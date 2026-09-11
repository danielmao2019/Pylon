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
├── RENDERERS  # Tuple[str, ...] = ("depth", "rgb", "segmentation", "normal_3d", "normal_2d"), every public point-cloud entry main and this branch both carry
├── POINT_SIZES  # Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0), odd and even so both kernel shapes are reached
├── RETURN_MASK_OPTIONS  # Tuple[bool, ...] = (False, True)
├── DEVICES  # Tuple[torch.device, ...] = cpu, followed by cuda:0 when cuda is available; the indexed spelling, since main's Camera compares a bare cuda unequal to its components' cuda:0
├── def build_point_cloud(scene: Dict[str, Any], device: torch.device) -> PointCloud
│   ├── # Rebuilds a scene's cloud from its stored tensors through the constructor both checkouts share, so main and this branch render the same points.
│   ├── calls PointCloud(xyz=scene["xyz"] on device, data={"rgb": scene["rgb"], "labels": scene["labels"], "normals": scene["normals"]} each on device)
│   └── return  # that cloud
├── def build_camera(scene: Dict[str, Any], camera_index: int, device: torch.device) -> Camera
│   ├── # Rebuilds one of a scene's cameras from its stored tensors through the constructors both checkouts share.
│   ├── impls camera_spec = scene["cameras"][camera_index]
│   ├── calls build_camera_intrinsics(model=scene["model"], params=camera_spec["params"], intr_convention=scene["intr_convention"], device=device)
│   ├── calls CameraExtrinsics(extrinsics=camera_spec["extrinsics"], extr_convention=scene["extr_convention"], device=device)
│   ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=device)
│   └── return  # that camera
└── def render_single_camera(renderer: str, pc: PointCloud, camera: Camera, resolution: Tuple[int, int], return_mask: bool, point_size: float) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ├── # Renders one camera through the named entry with the same keyword arguments on either checkout, each entry keeping its own default background.
    ├── assert renderer in RENDERERS  # f"{renderer=}"
    ├── if renderer == "depth"
    │   ├── calls render_depth_from_point_cloud(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── elif renderer == "rgb"
    │   ├── calls render_rgb_from_point_cloud(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── elif renderer == "segmentation"
    │   ├── calls render_segmentation_from_point_cloud(pc=pc, key="labels", camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    ├── elif renderer == "normal_3d"
    │   ├── calls render_normal_from_point_cloud_3d(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
    │   └── return  # its output
    └── else
        ├── calls render_normal_from_point_cloud_2d(pc=pc, camera=camera, resolution=resolution, return_mask=return_mask, point_size=point_size)
        └── return  # its output
```

`tasks/20260903_implement_batched_pc_render/render_on_main.py`

```text
render_on_main.py
├── import argparse
├── import torch
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
├── from scene_rendering import DEVICES, POINT_SIZES, RENDERERS, RETURN_MASK_OPTIONS, build_camera, build_point_cloud, render_single_camera
├── def main() -> None
│   ├── # Renders every scene with main's code in a child process launched inside the main checkout, so the branch has a fixed reference to compare against.
│   ├── impls args = the parsed --scenes_path and --output_path
│   ├── impls enable torch's deterministic algorithms  # main resolves a shared pixel by which write lands last, and deterministic mode makes that the last write in point order on cpu and cuda alike
│   ├── impls scenes = the scenes deserialized from args.scenes_path
│   ├── impls renders = an empty dict keyed by (device, scene name, camera index, renderer, point size, return_mask)
│   ├── for each device of DEVICES, scene, camera index, renderer, point size and return_mask
│   │   ├── calls build_point_cloud(scene=scene, device=device)
│   │   ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)
│   │   ├── calls render_single_camera(renderer=renderer, pc=pc, camera=camera, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)
│   │   └── impls renders[key] = that output moved to cpu
│   ├── impls kernels = an empty dict keyed by point size
│   ├── for each point size
│   │   ├── calls create_circular_kernel_offsets(point_size=point_size, device=torch.device("cpu"))
│   │   └── impls kernels[point size] = the offsets it returned
│   └── impls torch.save a dict of renders and kernels to args.output_path
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
├── impls REPO_ROOT = the repo root three levels above this file, inserted at the front of sys.path when absent so the repo-level imports below resolve
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud, render_depth_from_rendering_points
├── from scene_rendering import DEVICES, POINT_SIZES, RENDERERS, RETURN_MASK_OPTIONS, build_camera, build_point_cloud, render_single_camera
├── def main() -> None
│   ├── # Proves the two equivalences the task is done on: one camera handed over as a batch renders what main renders, and a batch renders what its cameras render one by one.
│   ├── impls args = the parsed --main_repo (a checkout of this repo's main) and --force
│   ├── impls main_repo = args.main_repo resolved to an absolute path  # it is the child's working directory and import root
│   ├── impls output_dir = this task's outputs/ directory
│   ├── impls set the CUBLAS_WORKSPACE_CONFIG environment variable to :4096:8, the workspace configuration deterministic cuBLAS requires
│   ├── impls enable torch's deterministic algorithms
│   ├── calls load_or_build_scenes(output_dir=output_dir, force=args.force)
│   ├── calls load_or_render_on_main(main_repo=main_repo, output_dir=output_dir, force=args.force)
│   ├── calls compare_single_camera_to_main(scenes=scenes, main_renders=main_renders)
│   ├── calls compare_batch_to_one_by_one(scenes=scenes)
│   ├── calls summarize_point_size_changes(main_renders=main_renders)
│   ├── impls report = the main and branch commits, the devices, that main rendered under deterministic algorithms, both comparison records with their required checks tallied, and the point-size summary  # main's scatter is racy on cuda otherwise, so its reference is the deterministic one
│   ├── impls write report as json to output_dir / "equivalence_report.json"
│   ├── impls print one summary line per tally
│   └── if any required check failed
│       └── raise SystemExit(1)
├── def load_or_build_scenes(output_dir: Path, force: bool) -> List[Dict[str, Any]]
│   ├── # Returns the fixed seeded scenes both checkouts render, from output_dir / "scenes.pt" unless it is missing or force asks for a rebuild.
│   ├── if the scenes file exists and not force
│   │   └── return  # the scenes torch.load reads from it
│   ├── calls build_scenes()
│   ├── impls torch.save those scenes to output_dir / "scenes.pt"
│   └── return  # those scenes
├── def build_scenes() -> List[Dict[str, Any]]
│   ├── # Builds scenes that reach every regime the batching changes: many points per pixel, few enough points for CUDA's small-matrix kernels, points culled by some cameras only, rescaled intrinsics, and all three pose conventions.
│   ├── impls generator = a torch.Generator seeded once
│   ├── impls collisions = twenty thousand points in a cube of side two about the origin, four opengl pinhole cameras on a sphere of radius four aimed at it, rendered at the (48, 64) the intrinsics state
│   ├── impls culling = two thousand points in a cube of side twelve, three opencv pinhole cameras on a radius-four sphere each seeing a different subset, intrinsics stated at (120, 160) but rendered at (60, 80), uint8 colours
│   ├── impls sparse = three hundred points in a cube of side one, three standard-convention pinhole cameras on a sphere of radius three, rendered at the (90, 120) the intrinsics state
│   ├── impls few_points = twenty-five points in a cube of side two about the origin, three opencv pinhole cameras on a sphere of radius four, rendered at the (30, 40) the intrinsics state  # few enough rows that CUDA picks its small-matrix kernels
│   ├── for each scene
│   │   ├── impls rgb, labels, normals = per-point colours, int64 labels below twenty, unit normals, all drawn from generator
│   │   └── impls cameras = one {params, extrinsics} per pose, each extrinsics an opencv look-at cam2world restated in the scene's convention by its fixed axis change
│   └── return  # the scenes as dicts of plain cpu tensors, names and conventions
├── def load_or_render_on_main(main_repo: Path, output_dir: Path, force: bool) -> Dict[str, Any]
│   ├── # Returns main's renders of the scenes, from output_dir / "main_renders.pt" unless it is missing, was rendered at another main commit or from other scenes, or force asks for a rerender.
│   ├── impls main_commit = the HEAD of main_repo, read through git
│   ├── impls scenes_digest = the sha256 of the scenes file's bytes
│   ├── assert main_commit equals the commit this repo's main branch points at  # the proof is against main as it stands, not an older checkout of it
│   ├── if the renders file exists and not force
│   │   ├── impls cached = the renders torch.load reads from it
│   │   └── if cached carries main_commit and scenes_digest
│   │       └── return cached
│   ├── impls run render_on_main.py by path under this interpreter with cwd main_repo and an environment carrying PYTHONPATH main_repo and CUBLAS_WORKSPACE_CONFIG, handing it the scenes path and the renders path, checked
│   ├── impls tag the saved renders with main_commit and scenes_digest
│   └── return  # those renders
├── def compare_single_camera_to_main(scenes: List[Dict[str, Any]], main_renders: Dict[str, Any]) -> List[Dict[str, Any]]
│   ├── # Compares every render this branch makes of one camera, handed over alone and, where the entry takes a batch, as a batch of one, with the one main made of the same scene, camera, renderer, point size and mask option.
│   ├── impls records = an empty list
│   ├── for each device of DEVICES, scene, camera index, renderer, point size and return_mask
│   │   ├── calls build_point_cloud(scene=scene, device=device)
│   │   ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)
│   │   ├── calls render_single_camera(renderer=renderer, pc=pc, camera=camera, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)
│   │   ├── calls compare_exactly(output=that output, reference=main's render of the same key)
│   │   ├── impls records gain a "camera" record carrying that comparison
│   │   └── if renderer == "depth"
│   │       ├── calls build_cameras(scene=scene, camera_indices=[camera_index], device=device)
│   │       ├── calls render_depth_from_point_cloud(pc=pc, camera=that batch of one, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)
│   │       ├── calls compare_exactly(output=its only slice, reference=main's render of the same key)
│   │       └── impls records gain a "batch_of_one" record carrying that comparison
│   ├── impls mark each record required when its point size is one  # above one pixel this branch's dilation grows a centred disc taking the nearest neighbour, and its depth entry applies it, where main did neither
│   └── return records
├── def compare_batch_to_one_by_one(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]
│   ├── # Compares, on this branch alone, one call over a scene's whole batch of cameras with each camera on its own, stage by stage, so the rounding CUDA's batched kernels introduce is told apart from a batching error.
│   ├── impls records = an empty list
│   ├── for each device of DEVICES and scene
│   │   ├── calls build_point_cloud(scene=scene, device=device)
│   │   ├── calls build_cameras(scene=scene, camera_indices=every camera index of the scene, device=device)
│   │   ├── for each num_divide of None and 2  # the second splits the points into chunks, so a chunk's own row count reaches the small-matrix kernels
│   │   │   ├── calls prepare_points_for_rendering(pc=pc, camera=that batch, resolution=scene["resolution"], num_divide=num_divide)
│   │   │   └── for each camera index
│   │   │       ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)
│   │   │       ├── calls prepare_points_for_rendering(pc=pc, camera=that camera, resolution=scene["resolution"], num_divide=num_divide)
│   │   │       ├── if device is cpu
│   │   │       │   └── calls compare_exactly(output=the rows valid keeps in that camera's slice of the batched points, with those rows' point indices, reference=that camera's points and original_data_indices)
│   │   │       ├── else
│   │   │       │   └── calls compare_preparations(output=the batched points and valid mask at that camera's slice, reference=that camera's points and original_data_indices, resolution=scene["resolution"])  # CUDA's batched inverse and product round unlike a single camera's
│   │   │       └── impls records gain a "prepare" record carrying that comparison and num_divide
│   │   ├── impls rendering_points, valid = the points and valid mask of the unchunked batched preparation  # one input handed to both sides, so the rasterizing stage is measured apart from the rounding before it
│   │   ├── for each return_mask
│   │   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=scene["resolution"], ignore_value=float("inf"), return_mask=return_mask, valid=valid)
│   │   │   └── for each camera index
│   │   │       ├── calls render_depth_from_rendering_points(rendering_points=that camera's slice of rendering_points, resolution=scene["resolution"], ignore_value=float("inf"), return_mask=return_mask, valid=that slice of valid)
│   │   │       ├── calls compare_exactly(output=the batched raster at that camera's slice, reference=that slice's own raster)
│   │   │       └── impls records gain a "rasterize" record carrying that comparison
│   │   ├── for each point size above one
│   │   │   ├── calls apply_point_size_postprocessing(rendered_image=the batched depth map rasterized above, depth_map=that same map, point_size=point_size, ignore_value=float("inf"))
│   │   │   └── for each camera index
│   │   │       ├── calls apply_point_size_postprocessing(rendered_image=that camera's slice of the batched depth map, depth_map=that same slice, point_size=point_size, ignore_value=float("inf"))
│   │   │       ├── calls compare_exactly(output=the batched dilation at that camera's slice, reference=that slice's own dilation)
│   │   │       └── impls records gain a "dilate" record carrying that comparison
│   │   └── for each point size and return_mask
│   │       ├── calls render_depth_from_point_cloud(pc=pc, camera=that batch, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)
│   │       └── for each camera index
│   │           ├── calls build_camera(scene=scene, camera_index=camera_index, device=device)
│   │           ├── calls render_depth_from_point_cloud(pc=pc, camera=that camera, resolution=scene["resolution"], return_mask=return_mask, point_size=point_size)
│   │           ├── calls compare_exactly(output=the batched map and mask at that camera's slice, reference=that camera's map and mask)
│   │           └── impls records gain a "depth" record carrying that comparison
│   ├── impls mark every record required but a cuda "depth" one  # end to end, cuda carries the preparation's rounding into the render, which the "prepare", "rasterize" and "dilate" records account for between them
│   └── return records
├── def build_cameras(scene: Dict[str, Any], camera_indices: List[int], device: torch.device) -> Cameras
│   ├── # Builds the batch of the named cameras of a scene, the one input only this branch's renderers take.
│   ├── calls build_camera_intrinsics(model=scene["model"], params=each param stacked over the named cameras, intr_convention=scene["intr_convention"], device=device)
│   ├── calls CameraExtrinsics(extrinsics=the named cameras' extrinsics stacked to [B, 4, 4], extr_convention=scene["extr_convention"], device=device)
│   ├── calls Cameras(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=device)
│   └── return  # that batch
├── def summarize_point_size_changes(main_renders: Dict[str, Any]) -> Dict[str, Any]
│   ├── # Records the two facts that account for every difference from main above one pixel: which point sizes grow a different disc on each side, and that main's depth-based entries ignore the point size altogether.
│   ├── impls summary = an empty dict
│   ├── for each point size
│   │   ├── calls create_circular_kernel_offsets(point_size=point_size, device=torch.device("cpu"))
│   │   └── impls summary gains whether that kernel and main's kernel for the same point size hold the same set of offsets
│   ├── for each of main's depth and normal_2d renders at every point size
│   │   ├── calls compare_exactly(output=that render, reference=main's render of the same device, scene, camera and mask option at point size one)
│   │   └── impls summary tallies whether that comparison came out equal
│   └── return summary
├── def compare_preparations(output: Tuple[torch.Tensor, torch.Tensor], reference: Tuple[torch.Tensor, torch.Tensor], resolution: Tuple[int, int]) -> Dict[str, Any]
│   ├── # Decides whether two preparations of one camera agree up to floating-point rounding, the test a cuda batch's preparation is held to.
│   ├── impls points, valid = output as cpu tensors
│   ├── impls reference_points, reference_indices = reference as cpu tensors  # the single camera's survivors and the points they are
│   ├── impls reference_valid = a mask over the slice's point axis, True at reference_indices
│   ├── impls tolerance = a few units in the last place of the points' dtype, relative to each coordinate's magnitude
│   ├── impls kept = valid & reference_valid
│   ├── impls points_close = every kept point's (x, y, depth) agrees with the reference row of that same point within tolerance  # a point either side culls lands on no pixel, so its coordinates carry nothing to compare
│   ├── impls flipped = the points where valid and reference_valid differ
│   ├── impls flips_explained = every flipped point lies within tolerance of the cull boundary it crossed, a depth of zero or an image edge of resolution, read off whichever side kept it
│   ├── calls compare_exactly(output=the rows valid keeps with their point indices, reference=reference)  # -> exact, so the report shows how often rounding moved anything at all
│   └── return  # {"equal": points_close and flips_explained, "exact": exact["equal"], "flipped_points": the count flipped marks, "max_abs_diff": exact["max_abs_diff"]}
├── def compare_exactly(output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], reference: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Dict[str, Any]
│   ├── # Decides whether two renders are the same result, which is the one test both equivalences are made of.
│   ├── impls output, reference = each argument as a tuple of cpu tensors, a lone map becoming a tuple of one
│   ├── impls equal = both tuples hold as many tensors, and each pair agrees in shape, in dtype and at every element, a NaN against a NaN counting as agreement
│   ├── impls differing_elements = for each pair of one shape, the count of elements that disagree that way
│   ├── impls nan_elements = for each pair, the count of positions where both hold NaN
│   ├── impls max_abs_diff = the largest absolute difference over positions both floating members of a pair hold finite
│   └── return  # {"equal": equal, "differing_elements": differing_elements, "nan_elements": nan_elements, "max_abs_diff": max_abs_diff}
└── if __name__ == "__main__"
    └── calls main()
```
