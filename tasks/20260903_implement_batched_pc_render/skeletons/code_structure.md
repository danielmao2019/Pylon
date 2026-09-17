# `tasks/20260903_implement_batched_pc_render/` code skeleton

## Code implementation structure

`tasks/20260903_implement_batched_pc_render/scene_rendering.py`

```text
scene_rendering.py
├── from typing import Any, Dict, Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud
├── from models.three_d.point_cloud.render.render_normal import render_normal_from_point_cloud_2d, render_normal_from_point_cloud_3d
├── from models.three_d.point_cloud.render.render_rgb import render_rgb_from_point_cloud
├── from models.three_d.point_cloud.render.render_segmentation import render_segmentation_from_point_cloud
├── RENDERERS  # Tuple[str, ...] = ("depth", "rgb", "segmentation", "normal_3d", "normal_2d"), every point-cloud entry main and this branch both carry that takes return_mask and point_size
├── POINT_SIZES  # Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0), odd and even so both kernel shapes are reached
├── RETURN_MASK_OPTIONS  # Tuple[bool, ...] = (False, True)
├── if torch's cuda is available
│   └── impls DEVICES = the cpu device followed by the cuda:0 device  # Tuple[torch.device, ...]; the indexed spelling, since main's Camera compares a bare cuda unequal to its components' cuda:0
├── else
│   └── impls DEVICES = the cpu device alone  # Tuple[torch.device, ...]
├── def build_point_cloud(scene: Dict[str, Any], device: torch.device) -> PointCloud
│   ├── # Rebuilds a scene's cloud from its stored tensors through the constructor both checkouts share, so main and this branch render the same points.
│   ├── calls PointCloud(xyz=scene["xyz"] on device, data={"rgb": scene["rgb"], "labels": scene["labels"], "normals": scene["normals"]} each on device)
│   └── return  # that cloud
├── def build_camera(scene: Dict[str, Any], camera_index: int, device: torch.device) -> Camera
│   ├── # Rebuilds one of a scene's cameras from its stored tensors through the constructors both checkouts share.
│   ├── impls camera_spec = scene["cameras"][camera_index]
│   ├── calls build_camera_intrinsics(model=scene["model"], params=camera_spec["params"], intr_convention=scene["intr_convention"], device=device)  # -> intrinsics
│   ├── calls CameraExtrinsics(extrinsics=camera_spec["extrinsics"], extr_convention=scene["extr_convention"], device=device)  # -> extrinsics
│   ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=device)
│   └── return  # that camera
└── def render_single_camera(renderer: str, pc: PointCloud, camera: Camera, resolution: Tuple[int, int], return_mask: bool, point_size: float) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ├── # Renders one camera through the named entry with the same keyword arguments on either checkout, each entry keeping its own default background.
    ├── def _validate_inputs [local]
    │   └── assert renderer is one of RENDERERS
    ├── calls _validate_inputs()
    ├── if renderer == "depth"
    │   ├── calls render_depth_from_point_cloud(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── if renderer == "rgb"
    │   ├── calls render_rgb_from_point_cloud(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── if renderer == "segmentation"
    │   ├── calls render_segmentation_from_point_cloud(pc=pc, key="labels", camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── if renderer == "normal_3d"
    │   ├── calls render_normal_from_point_cloud_3d(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── if renderer == "normal_2d"
    │   ├── calls render_normal_from_point_cloud_2d(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    └── assert 0, "Should not reach here."
```

`tasks/20260903_implement_batched_pc_render/render_on_main.py`

```text
render_on_main.py
├── import argparse
├── import torch
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
├── from scene_rendering import DEVICES, POINT_SIZES, RENDERERS, RETURN_MASK_OPTIONS, build_camera, build_point_cloud, render_single_camera
├── def main() -> None
│   ├── # Renders every scene with main's code in a child process launched inside the main checkout, so the branch has a fixed reference to compare against.
│   ├── impls parser = an argparse parser described as rendering the saved scenes with the checkout this process imports from
│   ├── impls add to parser a required str --scenes_path, the path of the saved scenes
│   ├── impls add to parser a required str --output_path, the path the renders are saved to
│   ├── impls args = the arguments parser parses
│   ├── impls enable torch's deterministic algorithms  # main resolves a shared pixel by which write lands last, and deterministic mode makes that the last write in point order on cpu and cuda alike
│   ├── impls scenes = the scenes deserialized from args.scenes_path
│   ├── impls renders = an empty dict keyed by (device name, scene name, camera index, renderer, point size, return_mask)
│   ├── for each device of DEVICES
│   │   └── for each scene of scenes
│   │       └── for each camera_index of the scene's camera indices
│   │           └── for each renderer of RENDERERS
│   │               └── for each point_size of POINT_SIZES
│   │                   └── for each return_mask of RETURN_MASK_OPTIONS
│   │                       ├── calls build_point_cloud(scene=scene, device=device)  # -> pc
│   │                       ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)  # -> camera
│   │                       ├── calls render_single_camera(renderer=renderer, pc=pc, camera=camera, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)  # -> output
│   │                       ├── if return_mask
│   │                       │   └── impls renders[(device name, scene name, camera_index, renderer, point_size, return_mask)] = (output[0] moved to cpu, output[1] moved to cpu)  # the map and its mask
│   │                       └── else
│   │                           └── impls renders[(device name, scene name, camera_index, renderer, point_size, return_mask)] = output moved to cpu
│   ├── impls kernels = an empty dict keyed by point size
│   ├── for each point_size of POINT_SIZES
│   │   └── calls create_circular_kernel_offsets(point_size=point_size, device=torch.device("cpu"))  # -> kernels[point_size]
│   ├── impls dilations = an empty dict keyed by (device name, scene name, camera index, point size)
│   ├── for each device of DEVICES
│   │   └── for each scene of scenes
│   │       └── for each camera_index of the scene's camera indices
│   │           └── for each point_size of POINT_SIZES
│   │               └── if point_size > 1.0
│   │                   ├── impls depth_map = renders at (device name, scene name, camera_index, "depth", 1.0, False) with the depth entry's own background of -1.0 set to positive infinity
│   │                   ├── calls apply_point_size_postprocessing(rendered_image=depth_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │                   └── impls dilations[(device name, scene name, camera_index, point_size)] = the dilation it returned
│   └── impls save a dict of renders, kernels and dilations to args.output_path through torch's save
└── if __name__ == "__main__"
    └── calls main()
```

`tasks/20260903_implement_batched_pc_render/prove_equivalence.py`

```text
prove_equivalence.py
├── import argparse
├── import hashlib
├── import json
├── import os
├── import subprocess
├── import sys
├── from pathlib import Path
├── from typing import Any, Dict, List, Tuple, Union
├── impls REPO_ROOT = the repo root three levels above this file
├── if REPO_ROOT is not on sys.path  # so the repo-level imports below resolve
│   └── impls put REPO_ROOT at the front of sys.path
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud, render_depth_from_rendering_points
├── from scene_rendering import DEVICES, POINT_SIZES, RENDERERS, RETURN_MASK_OPTIONS, build_camera, build_point_cloud, render_single_camera
├── def main() -> None
│   ├── # Proves the two equivalences the task is done on: one camera handed over as a batch renders what main renders, and a batch renders what its cameras render one by one.
│   ├── impls parser = an argparse parser described as proving single-camera renders equal main's and batched renders equal one-by-one renders
│   ├── impls add to parser a required Path --main_repo, a checkout of this repo's main branch
│   ├── impls add to parser a store-true --force, rebuilding the cached scenes and main renders
│   ├── impls args = the arguments parser parses
│   ├── impls main_repo = the main_repo argument resolved to an absolute path  # it is the child's working directory and import root
│   ├── impls output_dir = this task's outputs/ directory
│   ├── impls set the CUBLAS_WORKSPACE_CONFIG environment variable to :4096:8, the workspace configuration deterministic cuBLAS requires
│   ├── impls enable torch's deterministic algorithms
│   ├── calls load_or_build_scenes(output_dir=output_dir, force=args.force)  # -> scenes
│   ├── calls load_or_render_on_main(main_repo=main_repo, output_dir=output_dir, force=args.force)  # -> main_renders
│   ├── calls compare_single_camera_to_main(scenes=scenes, main_renders=main_renders)  # -> single_camera_records
│   ├── calls compare_batch_to_one_by_one(scenes=scenes)  # -> batch_records
│   ├── calls summarize_point_size_changes(main_renders=main_renders)  # -> point_size_summary
│   ├── calls summarize_tie_changes(single_camera_records=single_camera_records)  # -> tie_summary
│   ├── impls branch_worktree_clean = whether git status --porcelain run in REPO_ROOT prints nothing  # a report is evidence only for the commit it names
│   ├── impls comparisons = an empty dict keyed by comparison name
│   ├── for each name, records of ("dod_1", single_camera_records) and ("dod_2", batch_records)
│   │   ├── impls required_records = an empty list
│   │   ├── impls failures = an empty list
│   │   ├── for each record of records
│   │   │   └── if the record is required
│   │   │       ├── impls append record to required_records
│   │   │       └── if the record is not equal
│   │   │           └── impls append record to failures
│   │   └── impls comparisons[name] = records with their required tally: the count of required_records, that count less the count of failures as the equal ones, and failures
│   ├── impls report = the main commit main_renders carries, the branch HEAD read through git in REPO_ROOT, branch_worktree_clean, the device names, that main rendered under deterministic algorithms, every entry of comparisons, point_size_summary, and tie_summary  # main's scatter is racy on cuda otherwise, so its reference is the deterministic one
│   ├── impls write report as json indented by two to output_dir / "equivalence_report.json"
│   ├── impls print the commits, whether the branch worktree was clean, the devices and main's deterministic algorithms on one line
│   ├── for each name, comparison of comparisons
│   │   └── impls print name with its required equal count out of its total and its failure count
│   ├── for each label, entry of point_size_summary
│   │   └── impls print label with entry
│   ├── for each label, entry of tie_summary
│   │   └── impls print label with entry
│   ├── impls any_failed = False
│   ├── for each comparison of comparisons
│   │   └── if the comparison's required failures are non-empty
│   │       └── impls any_failed = True
│   └── if any_failed
│       └── raise SystemExit(1)
├── def load_or_build_scenes(output_dir: Path, force: bool) -> List[Dict[str, Any]]
│   ├── # Returns the fixed seeded scenes both checkouts render, from output_dir / "scenes.pt" unless it is missing or force asks for a rebuild.
│   ├── if the scenes file exists and not force
│   │   └── return  # the scenes torch's load reads from it
│   ├── calls build_scenes()  # -> scenes
│   ├── impls save scenes to output_dir / "scenes.pt" through torch's save
│   └── return scenes
├── def build_scenes() -> List[Dict[str, Any]]
│   ├── # Builds scenes that reach every regime the batching changes: many points per pixel, few enough points for CUDA's small-matrix kernels, points culled by some cameras only, rescaled intrinsics, all three pose conventions, and points that tie in depth on one pixel.
│   ├── impls generator = a torch random number generator seeded once with 0
│   ├── impls collisions = the "collisions" scene: twenty thousand points in a side-two cube about the origin, float32 colours, four opengl pinhole cameras on a radius-four sphere with focal 60, rendered at the (48, 64) the standard intrinsics state
│   ├── impls culling = the "culling" scene: two thousand points in a side-twelve cube about the origin, uint8 colours, three opencv pinhole cameras on a radius-four sphere with focal 150, standard intrinsics stated at (120, 160) but rendered at (60, 80)  # each camera sees a different subset
│   ├── impls sparse = the "sparse" scene: three hundred points in a side-one cube about the origin, float32 colours, three standard-convention pinhole cameras on a radius-three sphere with focal 240, rendered at the (90, 120) the standard intrinsics state
│   ├── impls few_points = the "few_points" scene: twelve points in a side-two cube about the origin, float32 colours, three opencv pinhole cameras on a radius-four sphere with focal 37.5, rendered at the (30, 40) the standard intrinsics state  # at most sixteen rows, the most CUDA's small-matrix kernel takes
│   ├── impls ties = the "ties" scene: twelve points in six coincident pairs in a side-two cube about the origin, float32 colours, three opencv pinhole cameras on a radius-four sphere with focal 60, rendered at the (48, 64) the standard intrinsics state  # the two points of a pair share a depth and a pixel from every camera
│   ├── impls axis_changes = per pose convention, the fixed rotation that right-multiplies an opencv camera-to-world rotation into the same pose in that convention  # opencv the identity, opengl diag(1, -1, -1), standard [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
│   ├── for each scene of collisions, culling, sparse, few_points and ties  # each point draws its own colour, label and normal, the two points of a tied pair included, so a render shows which point of a pair it kept
│   │   ├── impls num_points, rgb_dtype = the scene's point count, and the colour dtype popped from it
│   │   ├── if rgb_dtype is uint8
│   │   │   └── impls scene["rgb"] = uint8 colours in [0, 255] drawn from generator
│   │   ├── else
│   │   │   └── impls scene["rgb"] = float32 colours in [0, 1) drawn from generator
│   │   ├── impls scene["labels"], scene["normals"] = int64 labels below twenty and unit normals, both drawn from generator
│   │   ├── impls num_cameras = the camera count popped from the scene
│   │   ├── impls azimuth = [num_cameras] azimuths evenly spaced over a full turn, starting from 0.3 rad
│   │   ├── impls elevation = [num_cameras] elevations alternating +0.35 and -0.35 rad, starting at +0.35
│   │   ├── impls centre = [num_cameras, 3] camera centres at azimuth and elevation on a sphere about the origin of the radius popped from the scene
│   │   ├── impls forward = the unit vectors from centre towards the origin  # +Z of each opencv look-at pose at centre aimed at the origin
│   │   ├── impls right = forward x world +Z, normalized to unit length  # +X
│   │   ├── impls down = forward x right  # +Y
│   │   ├── impls cam2world = [num_cameras, 4, 4] identity matrices
│   │   ├── impls cam2world's rotation blocks = the columns (right, down, forward), right-multiplied by axis_changes of the scene's extr_convention  # the opencv rotations restated in the scene's convention
│   │   ├── impls cam2world's translations = centre
│   │   ├── impls focal, stated_height, stated_width = the base focal length and the resolution the intrinsics state, both popped from the scene
│   │   ├── impls scene["cameras"] = an empty list
│   │   └── for each camera_index below num_cameras
│   │       └── impls append to scene["cameras"] {"params": 0-dim tensors fx = focal * (1 + 0.1 * camera_index), fy = focal * (1.05 + 0.1 * camera_index), cx = stated_width / 2 + camera_index, cy = stated_height / 2 - camera_index, h = stated_height, w = stated_width; "extrinsics": a copy of cam2world[camera_index]}
│   └── return  # [collisions, culling, sparse, few_points, ties], each now a dict of its name, model, conventions, resolution and cpu tensors
├── def load_or_render_on_main(main_repo: Path, output_dir: Path, force: bool) -> Dict[str, Any]
│   ├── # Returns main's renders of the scenes, from output_dir / "main_renders.pt" unless it is missing, was rendered at another main commit or from other scenes, or force asks for a rerender.
│   ├── impls main_commit = the HEAD of main_repo, read through git
│   ├── impls scenes_digest = the hex sha256 of the scenes file's bytes
│   ├── impls main_branch_commit = the commit this repo's main branch points at, read through git in REPO_ROOT
│   ├── assert main_commit == main_branch_commit  # the proof is against main as it stands, not an older checkout of it
│   ├── if the renders file exists and not force
│   │   ├── impls cached = the renders torch's load reads from it
│   │   └── if cached carries main_commit and scenes_digest
│   │       └── return cached
│   ├── impls run render_on_main.py by path under this interpreter with cwd main_repo and the current environment plus PYTHONPATH=main_repo and CUBLAS_WORKSPACE_CONFIG=:4096:8, handing it the scenes path and the renders path, checked
│   ├── impls main_renders = the renders torch's load reads from the renders file
│   ├── impls add main_commit and scenes_digest to main_renders
│   ├── impls save main_renders back to the renders file through torch's save
│   └── return main_renders
├── def compare_single_camera_to_main(scenes: List[Dict[str, Any]], main_renders: Dict[str, Any]) -> List[Dict[str, Any]]
│   ├── # Compares every render this branch makes of one camera, handed over alone and, where the entry takes a batch, as a batch of one, with the one main made of the same scene, camera, renderer, point size and mask option.
│   ├── impls records = an empty list
│   ├── for each device of DEVICES
│   │   └── for each scene of scenes
│   │       └── for each camera_index of the scene's camera indices
│   │           └── for each renderer of RENDERERS
│   │               └── for each point_size of POINT_SIZES
│   │                   └── for each return_mask of RETURN_MASK_OPTIONS
│   │                       ├── calls build_point_cloud(scene=scene, device=device)  # -> pc
│   │                       ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)  # -> camera
│   │                       ├── calls render_single_camera(renderer=renderer, pc=pc, camera=camera, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)  # -> output
│   │                       ├── impls main_output = main's render at (device name, scene name, camera_index, renderer, point_size, return_mask)
│   │                       ├── calls compare_exactly(output=output, reference=main_output)  # -> comparison
│   │                       ├── impls append to records a "camera" record of the device name, scene name, camera_index, renderer, point_size and return_mask, merged with comparison
│   │                       └── if renderer == "depth"
│   │                           ├── calls build_cameras(scene=scene, camera_indices=[camera_index], device=device)  # -> cameras
│   │                           ├── calls render_depth_from_point_cloud(pc=pc, camera=cameras, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)  # -> batch_output
│   │                           ├── if return_mask
│   │                           │   └── impls batch_slice = (batch_output[0][0], batch_output[1][0])  # the map's and the mask's only slice
│   │                           ├── else
│   │                           │   └── impls batch_slice = batch_output[0]
│   │                           ├── calls compare_exactly(output=batch_slice, reference=main_output)  # -> comparison
│   │                           └── impls append to records a "batch_of_one" record of the device name, scene name, camera_index, renderer, point_size and return_mask, merged with comparison
│   ├── for each record of records  # above one pixel this branch's dilation grows a centred disc taking the nearest neighbour, and its depth entry applies it, where main did neither; at a tied depth this branch keeps the lowest point index, where main keeps the last point in order on cpu and the first on cuda
│   │   └── impls mark the record required when its point size is one and its scene is not "ties"
│   └── return records
├── def compare_batch_to_one_by_one(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]
│   ├── # Compares, on this branch alone, one call over a scene's whole batch of cameras with each camera on its own, stage by stage, so the rounding CUDA's batched kernels introduce is told apart from a batching error.
│   ├── impls records = an empty list
│   ├── for each device of DEVICES
│   │   └── for each scene of scenes
│   │       ├── calls build_point_cloud(scene=scene, device=device)  # -> pc
│   │       ├── calls build_cameras(scene=scene, camera_indices=every camera index of the scene, device=device)  # -> cameras
│   │       ├── impls batch_preparations = an empty dict keyed by num_divide
│   │       ├── for each num_divide of None and 2  # the second splits the points into chunks, so a chunk's own row count reaches the small-matrix kernels
│   │       │   ├── calls prepare_points_for_rendering(pc=pc, camera=cameras, resolution=scene["resolution"], num_divide=num_divide)  # -> batch_preparations[num_divide]
│   │       │   ├── impls batch_points, batch_valid = the points and valid mask of batch_preparations[num_divide]
│   │       │   └── for each camera_index of the scene's camera indices
│   │       │       ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)  # -> camera
│   │       │       ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=scene["resolution"], num_divide=num_divide)  # -> points, _, original_data_indices
│   │       │       ├── if device is cpu
│   │       │       │   └── calls compare_exactly(output=the rows batch_valid keeps in that camera's slice of batch_points, with those rows' point indices, reference=(points, original_data_indices))  # -> comparison
│   │       │       ├── else
│   │       │       │   └── calls compare_preparations(output=(batch_points[camera_index], batch_valid[camera_index]), reference=(points, original_data_indices), pc=pc, camera=camera, resolution=scene["resolution"])  # -> comparison; CUDA's batched inverse and product round unlike a single camera's
│   │       │       └── impls append to records a "prepare" record of the device name, scene name, camera_index and num_divide, merged with comparison
│   │       ├── impls rendering_points, valid = the points and valid mask of batch_preparations[None], the unchunked one  # one input handed to both sides, so the rasterizing stage is measured apart from the rounding before it
│   │       ├── for each return_mask of RETURN_MASK_OPTIONS
│   │       │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=scene["resolution"], ignore_value=float("inf"), return_mask=return_mask, valid=valid)  # -> batch_raster
│   │       │   ├── if return_mask  # the depth map of batch_raster
│   │       │   │   └── impls batch_depth_map = batch_raster[0]
│   │       │   ├── else
│   │       │   │   └── impls batch_depth_map = batch_raster
│   │       │   └── for each camera_index of the scene's camera indices
│   │       │       ├── calls render_depth_from_rendering_points(rendering_points=rendering_points[camera_index], resolution=scene["resolution"], ignore_value=float("inf"), return_mask=return_mask, valid=valid[camera_index])  # -> slice_raster
│   │       │       ├── if return_mask
│   │       │       │   └── impls batch_slice = (batch_raster[0][camera_index], batch_raster[1][camera_index])  # the map's and the mask's slice at camera_index
│   │       │       ├── else
│   │       │       │   └── impls batch_slice = batch_raster[camera_index]
│   │       │       ├── calls compare_exactly(output=batch_slice, reference=slice_raster)  # -> comparison
│   │       │       └── impls append to records a "rasterize" record of the device name, scene name, camera_index and return_mask, merged with comparison
│   │       ├── for each point_size of POINT_SIZES
│   │       │   └── if point_size > 1.0
│   │       │       ├── calls apply_point_size_postprocessing(rendered_image=batch_depth_map, depth_map=batch_depth_map, point_size=point_size, ignore_value=float("inf"))  # -> batch_dilation
│   │       │       └── for each camera_index of the scene's camera indices
│   │       │           ├── calls apply_point_size_postprocessing(rendered_image=batch_depth_map[camera_index], depth_map=batch_depth_map[camera_index], point_size=point_size, ignore_value=float("inf"))  # -> slice_dilation
│   │       │           ├── calls compare_exactly(output=batch_dilation[camera_index], reference=slice_dilation)  # -> comparison
│   │       │           └── impls append to records a "dilate" record of the device name, scene name, camera_index and point_size, merged with comparison
│   │       └── for each point_size of POINT_SIZES
│   │           └── for each return_mask of RETURN_MASK_OPTIONS
│   │               ├── calls render_depth_from_point_cloud(pc=pc, camera=cameras, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)  # -> batch_output
│   │               └── for each camera_index of the scene's camera indices
│   │                   ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)  # -> camera
│   │                   ├── calls render_depth_from_point_cloud(pc=pc, camera=camera, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)  # -> camera_output
│   │                   ├── if return_mask
│   │                   │   └── impls batch_slice = (batch_output[0][camera_index], batch_output[1][camera_index])  # the map's and the mask's slice at camera_index
│   │                   ├── else
│   │                   │   └── impls batch_slice = batch_output[camera_index]
│   │                   ├── calls compare_exactly(output=batch_slice, reference=camera_output)  # -> comparison
│   │                   └── impls append to records a "depth" record of the device name, scene name, camera_index, point_size and return_mask, merged with comparison
│   ├── for each record of records  # end to end, cuda carries the preparation's rounding into the render, which the "prepare", "rasterize" and "dilate" records account for between them
│   │   └── impls mark the record required unless it is a cuda "depth" one
│   └── return records
├── def build_cameras(scene: Dict[str, Any], camera_indices: List[int], device: torch.device) -> Cameras
│   ├── # Builds the batch of the named cameras of a scene, the one input only this branch's renderers take.
│   ├── impls params = an empty dict
│   ├── for each key of the first named camera's params
│   │   ├── impls param_values = an empty list
│   │   ├── for each camera_index of camera_indices
│   │   │   └── impls append scene["cameras"][camera_index]["params"][key] to param_values
│   │   └── impls params[key] = param_values stacked
│   ├── calls build_camera_intrinsics(model=scene["model"], params=params, intr_convention=scene["intr_convention"], device=device)  # -> intrinsics
│   ├── impls extrinsics_list = an empty list
│   ├── for each camera_index of camera_indices
│   │   └── impls append scene["cameras"][camera_index]["extrinsics"] to extrinsics_list
│   ├── impls extrinsics_matrices = extrinsics_list stacked to [B, 4, 4]
│   ├── calls CameraExtrinsics(extrinsics=extrinsics_matrices, extr_convention=scene["extr_convention"], device=device)  # -> extrinsics
│   ├── calls Cameras(intrinsics=intrinsics, extrinsics=extrinsics, device=device)
│   └── return  # that batch
├── def summarize_point_size_changes(main_renders: Dict[str, Any]) -> Dict[str, Any]
│   ├── # Records the three facts that account for every difference from main above one pixel: which point sizes grow a different disc on each side, that main's depth-based entries ignore the point size altogether, and which neighbour each side's dilation keeps on one depth map.
│   ├── impls summary = an empty dict
│   ├── for each point_size of POINT_SIZES
│   │   ├── calls create_circular_kernel_offsets(point_size=point_size, device=torch.device("cpu"))  # -> kernel_offsets
│   │   └── impls add to summary, under point_size, whether kernel_offsets and main's kernel offsets for point_size hold the same set of (y, x) offsets
│   ├── for each (device name, scene name, camera_index, renderer, point_size, return_mask) key of main's renders and the render under it
│   │   ├── if renderer is neither "depth" nor "normal_2d"
│   │   │   └── continue
│   │   ├── calls compare_exactly(output=render, reference=main's render of the same device, scene, camera and mask option at point size one)  # -> comparison
│   │   └── impls summary tallies, under renderer and point_size, whether comparison came out equal
│   ├── for each (device name, scene name, camera_index, point_size) key of main's dilations, every point size above one, and the main_dilation under it
│   │   ├── impls depth_map = main's depth render of that device, scene and camera at point size one without a mask, the depth entry's own background of -1.0 set to positive infinity  # the map render_on_main dilated with main's own dilation
│   │   ├── calls apply_point_size_postprocessing(rendered_image=depth_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))  # -> dilation
│   │   ├── calls compare_exactly(output=dilation, reference=main_dilation)  # -> comparison
│   │   └── impls summary tallies, under point_size, whether comparison came out equal  # main keeps the last nearer neighbour in kernel order, this branch the nearest
│   └── return summary
├── def summarize_tie_changes(single_camera_records: List[Dict[str, Any]]) -> Dict[str, Any]
│   ├── # Records, per device and renderer, how many of the ties scene's point-size-one renders come out equal to main's, the one regime where main's choice between tied points depends on the device.
│   ├── impls summary = an empty dict
│   ├── for each record of single_camera_records
│   │   └── if the record's scene name is "ties" and its point size is 1.0
│   │       └── impls summary tallies, under the record's device name and renderer, whether the record came out equal
│   └── return summary
├── def compare_preparations(output: Tuple[torch.Tensor, torch.Tensor], reference: Tuple[torch.Tensor, torch.Tensor], pc: PointCloud, camera: Camera, resolution: Tuple[int, int]) -> Dict[str, Any]
│   ├── # Decides whether two preparations of one camera agree up to floating-point rounding, the test a cuda batch's preparation is held to.
│   ├── impls points, valid = output as cpu tensors
│   ├── impls reference_points, reference_indices = reference as cpu tensors  # the single camera's survivors and the points they are
│   ├── impls reference_valid = a mask over the slice's point axis, True at reference_indices
│   ├── impls reference_rows = an [N, 3] zero tensor like points holding reference_points at the rows reference_indices name  # each survivor at the row of the point it is
│   ├── impls magnitude = the larger of the norm of camera's centre and the largest coordinate magnitude in pc  # the size of the numbers the world-to-camera transform rounds
│   ├── calls camera.scale_intrinsics(resolution=resolution)  # -> render_camera, whose focal lengths are the ones the preparation projects with at resolution
│   ├── impls camera_frame_tolerance = 4 times the machine epsilon of points' dtype times magnitude  # a few units in the last place of the points' dtype
│   ├── impls depth = per point, its depth in points at the points valid keeps and its depth in reference_rows at the rest  # read off whichever side kept it
│   ├── impls tolerance = [N, 3] per point: camera_frame_tolerance * render_camera's fx / depth for x, camera_frame_tolerance * render_camera's fy / depth for y, and camera_frame_tolerance for depth  # the projection multiplies camera-frame rounding by focal length over depth
│   ├── impls kept = valid & reference_valid
│   ├── impls points_close = every kept point's (x, y, depth) row of points agrees with its row of reference_rows within its row of tolerance  # a point either side culls lands on no pixel, so its coordinates carry nothing to compare
│   ├── impls flipped = the points where valid and reference_valid differ
│   ├── impls flipped_rows = per flipped point, its row of points where valid keeps it and its row of reference_rows where valid culls it  # the side that culled it keeps no row, so which boundary it crossed is not on record
│   ├── impls flips_explained = every flipped row has |x| or |x - W| within its x tolerance, |y| or |y - H| within its y tolerance, or |depth| within its depth tolerance, (H, W) being resolution  # some cull boundary: a depth of zero or an image edge of resolution
│   ├── calls compare_exactly(output=the rows valid keeps with their point indices, reference=reference)  # -> exact, so the report shows how often rounding moved anything at all
│   └── return  # {"equal": points_close and flips_explained, "exact": exact["equal"], "flipped_points": the count flipped marks, "max_abs_diff": exact["max_abs_diff"]}
├── def compare_exactly(output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], reference: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Dict[str, Any]
│   ├── # Decides whether two renders are the same result, which is the one test both equivalences are made of.
│   ├── def _normalize_inputs [local]
│   │   ├── if output is not a tuple  # a lone map becoming a tuple of one
│   │   │   └── impls output = a tuple of output alone
│   │   ├── if reference is not a tuple
│   │   │   └── impls reference = a tuple of reference alone
│   │   ├── impls cpu_members = an empty list
│   │   ├── for each member of output
│   │   │   └── impls append to cpu_members member moved to cpu
│   │   ├── impls output = cpu_members as a tuple
│   │   ├── impls cpu_reference_members = an empty list
│   │   ├── for each member of reference
│   │   │   └── impls append to cpu_reference_members member moved to cpu
│   │   ├── impls reference = cpu_reference_members as a tuple
│   │   └── return output, reference
│   ├── calls _normalize_inputs(output=output, reference=reference)  # -> output, reference
│   ├── if output and reference hold as many tensors
│   │   └── impls pairs = output and reference paired in order
│   ├── else
│   │   └── impls pairs = an empty list
│   ├── impls disagreements = an empty list
│   ├── for each member, reference_member of pairs
│   │   ├── if member and reference_member share a shape
│   │   │   └── impls append to disagreements the elements where member and reference_member differ and are not both NaN
│   │   └── else
│   │       └── impls append None to disagreements
│   ├── impls equal = whether output and reference hold as many tensors
│   ├── for each (member, reference_member), disagreement of pairs paired with disagreements
│   │   └── impls equal = equal and member and reference_member share a shape and a dtype and disagreement marks no element
│   ├── impls differing_elements = an empty list  # None for a pair of two shapes
│   ├── for each disagreement of disagreements
│   │   ├── if disagreement is None
│   │   │   └── impls append None to differing_elements
│   │   └── else
│   │       └── impls append the count of elements disagreement marks to differing_elements
│   ├── impls nan_elements = an empty list  # None for a pair of two shapes
│   ├── for each member, reference_member of pairs
│   │   ├── if member and reference_member share a shape
│   │   │   └── impls append to nan_elements the count of positions where both hold NaN
│   │   └── else
│   │       └── impls append None to nan_elements
│   ├── impls max_abs_diff = None  # None when no pair is of one shape with both members floating
│   ├── for each member, reference_member of pairs
│   │   └── if member and reference_member share a shape and both are floating
│   │       └── impls max_abs_diff = the larger of max_abs_diff and the largest absolute difference over positions both hold finite
│   └── return  # {"equal": equal, "differing_elements": differing_elements, "nan_elements": nan_elements, "max_abs_diff": max_abs_diff}
└── if __name__ == "__main__"
    └── calls main()
```
