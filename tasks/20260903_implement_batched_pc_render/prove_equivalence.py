import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
# So the repo-level imports below resolve.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from scene_rendering import (
    DEVICES,
    POINT_SIZES,
    RENDERERS,
    RETURN_MASK_OPTIONS,
    build_camera,
    build_point_cloud,
    render_single_camera,
)

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import (
    apply_point_size_postprocessing,
)
from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import (
    create_circular_kernel_offsets,
)
from models.three_d.point_cloud.render.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from models.three_d.point_cloud.render.render_depth import (
    render_depth_from_point_cloud,
    render_depth_from_rendering_points,
)


def main() -> None:
    """Proves the two equivalences the task is done on: one camera handed over as a batch renders what main renders, and a batch renders what its cameras render one by one.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Prove single-camera renders equal main's and batched renders equal one-by-one renders."
    )
    parser.add_argument(
        "--main_repo",
        type=Path,
        required=True,
        help="Path of a checkout of this repo's main branch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached scenes and main renders.",
    )
    args = parser.parse_args()
    # It is the child's working directory and import root.
    main_repo = args.main_repo.resolve()
    output_dir = Path(__file__).resolve().parent / "outputs"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    scenes = load_or_build_scenes(output_dir=output_dir, force=args.force)
    main_renders = load_or_render_on_main(
        main_repo=main_repo, output_dir=output_dir, force=args.force
    )
    single_camera_records = compare_single_camera_to_main(
        scenes=scenes, main_renders=main_renders
    )
    batch_records = compare_batch_to_one_by_one(scenes=scenes)
    point_size_summary = summarize_point_size_changes(main_renders=main_renders)
    tie_summary = summarize_tie_changes(single_camera_records=single_camera_records)
    # A report is evidence only for the commit it names.
    branch_worktree_clean = (
        subprocess.run(
            args=["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        == ""
    )

    # --- Both record lists with their required checks tallied
    comparisons = {}
    for name, records in (("dod_1", single_camera_records), ("dod_2", batch_records)):
        required_records = []
        failures = []
        for record in records:
            if record["required"]:
                required_records.append(record)
                if not record["equal"]:
                    failures.append(record)
        comparisons[name] = {
            "required": {
                "total": len(required_records),
                "equal": len(required_records) - len(failures),
                "failures": failures,
            },
            "records": records,
        }

    # --- Report: the commits, whether the branch worktree was clean, the devices, both comparisons, the point-size summary and the tie summary
    report = {
        "main_commit": main_renders["main_commit"],
        "branch_commit": subprocess.run(
            args=["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "branch_worktree_clean": branch_worktree_clean,
        "devices": list(map(str, DEVICES)),
        # main's scatter is racy on cuda otherwise, so its reference is the deterministic one render_on_main.py switches on before it renders.
        "main_deterministic_algorithms": True,
        **comparisons,
        "point_size_summary": point_size_summary,
        "tie_summary": tie_summary,
    }
    (output_dir / "equivalence_report.json").write_text(json.dumps(report, indent=2))

    # --- One summary line per tally
    print(
        f"commits: main {report['main_commit']}, branch {report['branch_commit']}, "
        f"branch worktree clean {report['branch_worktree_clean']}, "
        f"devices {report['devices']}, "
        f"main deterministic algorithms {report['main_deterministic_algorithms']}"
    )
    for name, comparison in comparisons.items():
        print(
            f"{name} required: {comparison['required']['equal']}/"
            f"{comparison['required']['total']} equal, "
            f"{len(comparison['required']['failures'])} failures"
        )
    for label, entry in point_size_summary.items():
        print(f"point-size summary, {label}: {entry}")
    for label, entry in tie_summary.items():
        print(f"tie summary, {label}: {entry}")

    # --- Exit non-zero when any required check failed
    any_failed = False
    for comparison in comparisons.values():
        if comparison["required"]["failures"]:
            any_failed = True
    if any_failed:
        raise SystemExit(1)


def load_or_build_scenes(output_dir: Path, force: bool) -> List[Dict[str, Any]]:
    """Returns the fixed seeded scenes both checkouts render, from output_dir / "scenes.pt" unless it is missing or force asks for a rebuild.

    Args:
        output_dir: This task's outputs/ directory.
        force: Whether to rebuild the scenes even when "scenes.pt" exists.

    Returns:
        The scene dicts build_scenes returns.
    """
    if (output_dir / "scenes.pt").exists() and not force:
        return torch.load(output_dir / "scenes.pt")
    scenes = build_scenes()
    torch.save(scenes, output_dir / "scenes.pt")
    return scenes


def build_scenes() -> List[Dict[str, Any]]:
    """Builds scenes that reach every regime the batching changes: many points per pixel, few enough points for CUDA's small-matrix kernels, points culled by some cameras only, rescaled intrinsics, all three pose conventions, and points that tie in depth on one pixel.

    Args:
        None.

    Returns:
        The scene dicts, every tensor on cpu: "name"; the cloud as "xyz" [N, 3] float32 world-space points, "rgb" [N, 3] colours (float32 in [0, 1], or uint8 in [0, 255]), "labels" [N] int64 labels below twenty and "normals" [N, 3] float32 world-space unit normals; the intrinsics "model" ("pinhole") and "intr_convention" ("standard"); the pose "extr_convention" ("opengl", "opencv" or "standard"); the render "resolution" as (H, W); and "cameras", a list of {"params": 0-dim float32 fx / fy / cx / cy / h / w, "extrinsics": [4, 4] float32 camera-to-world in the scene's extr_convention}.
    """
    generator = torch.Generator().manual_seed(0)

    # --- The five scenes: a seeded cloud about the origin, how many cameras sit on a sphere of what radius around it, their base focal length, and the resolution the intrinsics state versus the one rendered
    collisions = {
        "name": "collisions",
        "xyz": torch.rand(20000, 3, generator=generator) * 2.0 - 1.0,
        "rgb_dtype": torch.float32,
        "model": "pinhole",
        "intr_convention": "standard",
        "extr_convention": "opengl",
        "num_cameras": 4,
        "radius": 4.0,
        "focal": 60.0,
        "stated_resolution": (48, 64),
        "resolution": (48, 64),
    }
    # Each camera sees a different subset.
    culling = {
        "name": "culling",
        "xyz": torch.rand(2000, 3, generator=generator) * 12.0 - 6.0,
        "rgb_dtype": torch.uint8,
        "model": "pinhole",
        "intr_convention": "standard",
        "extr_convention": "opencv",
        "num_cameras": 3,
        "radius": 4.0,
        "focal": 150.0,
        "stated_resolution": (120, 160),
        "resolution": (60, 80),
    }
    sparse = {
        "name": "sparse",
        "xyz": torch.rand(300, 3, generator=generator) - 0.5,
        "rgb_dtype": torch.float32,
        "model": "pinhole",
        "intr_convention": "standard",
        "extr_convention": "standard",
        "num_cameras": 3,
        "radius": 3.0,
        "focal": 240.0,
        "stated_resolution": (90, 120),
        "resolution": (90, 120),
    }
    # At most sixteen rows, the most CUDA's small-matrix kernel takes.
    few_points = {
        "name": "few_points",
        "xyz": torch.rand(12, 3, generator=generator) * 2.0 - 1.0,
        "rgb_dtype": torch.float32,
        "model": "pinhole",
        "intr_convention": "standard",
        "extr_convention": "opencv",
        "num_cameras": 3,
        "radius": 4.0,
        "focal": 37.5,
        "stated_resolution": (30, 40),
        "resolution": (30, 40),
    }
    # The two points of a pair share a depth and a pixel from every camera.
    ties = {
        "name": "ties",
        "xyz": (torch.rand(6, 3, generator=generator) * 2.0 - 1.0).repeat_interleave(
            2, dim=0
        ),
        "rgb_dtype": torch.float32,
        "model": "pinhole",
        "intr_convention": "standard",
        "extr_convention": "opencv",
        "num_cameras": 3,
        "radius": 4.0,
        "focal": 60.0,
        "stated_resolution": (48, 64),
        "resolution": (48, 64),
    }

    # The fixed axis change per pose convention: an opencv camera-to-world rotation right-multiplied by it is the same pose in that convention.
    axis_changes = {
        "opencv": torch.eye(3),
        "opengl": torch.diag(torch.tensor([1.0, -1.0, -1.0])),
        "standard": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    }
    # Each point draws its own colour, label and normal, the two points of a tied pair included, so a render shows which point of a pair it kept.
    for scene in (collisions, culling, sparse, few_points, ties):
        # --- Per-point colours, labels and normals
        num_points, rgb_dtype = scene["xyz"].shape[0], scene.pop("rgb_dtype")
        if rgb_dtype == torch.uint8:
            scene["rgb"] = torch.randint(
                low=0,
                high=256,
                size=(num_points, 3),
                generator=generator,
                dtype=torch.uint8,
            )
        else:
            scene["rgb"] = torch.rand(num_points, 3, generator=generator)
        scene["labels"], scene["normals"] = (
            torch.randint(
                low=0,
                high=20,
                size=(num_points,),
                generator=generator,
                dtype=torch.int64,
            ),
            torch.nn.functional.normalize(
                torch.randn(num_points, 3, generator=generator), dim=-1
            ),
        )

        # --- Cameras: opencv look-at poses aimed at the origin (+Z forward, +Y image-down), restated in the scene's convention, each with its own intrinsics
        num_cameras = scene.pop("num_cameras")
        azimuth = torch.arange(num_cameras) * (2.0 * torch.pi / num_cameras) + 0.3
        elevation = 0.35 * (1.0 - 2.0 * (torch.arange(num_cameras) % 2))
        centre = scene.pop("radius") * torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        forward = -centre / centre.norm(dim=-1, keepdim=True)
        right = torch.nn.functional.normalize(
            torch.linalg.cross(
                forward, torch.tensor([0.0, 0.0, 1.0]).expand_as(forward)
            ),
            dim=-1,
        )
        down = torch.linalg.cross(forward, right)
        cam2world = torch.eye(4).repeat(num_cameras, 1, 1)
        cam2world[:, :3, :3] = (
            torch.stack([right, down, forward], dim=-1)
            @ axis_changes[scene["extr_convention"]]
        )
        cam2world[:, :3, 3] = centre
        focal, (stated_height, stated_width) = scene.pop("focal"), scene.pop(
            "stated_resolution"
        )
        scene["cameras"] = []
        for camera_index in range(num_cameras):
            scene["cameras"].append(
                {
                    "params": {
                        "fx": torch.tensor(focal * (1.0 + 0.1 * camera_index)),
                        "fy": torch.tensor(focal * (1.05 + 0.1 * camera_index)),
                        "cx": torch.tensor(stated_width / 2.0 + camera_index),
                        "cy": torch.tensor(stated_height / 2.0 - camera_index),
                        "h": torch.tensor(float(stated_height)),
                        "w": torch.tensor(float(stated_width)),
                    },
                    "extrinsics": cam2world[camera_index].clone(),
                }
            )
    return [collisions, culling, sparse, few_points, ties]


def load_or_render_on_main(
    main_repo: Path, output_dir: Path, force: bool
) -> Dict[str, Any]:
    """Returns main's renders of the scenes, from output_dir / "main_renders.pt" unless it is missing, was rendered at another main commit or from other scenes, or force asks for a rerender.

    Args:
        main_repo: Absolute path of a checkout of this repo's main branch.
        output_dir: This task's outputs/ directory, already holding "scenes.pt".
        force: Whether to rerender even when "main_renders.pt" exists at main's commit and from these scenes.

    Returns:
        The dict of "renders", mapping (device name, scene name, camera index, renderer, point size, return_mask) to main's cpu render (the map, or the (map, [H, W] bool mask) tuple when return_mask is True); "kernels", mapping each point size to main's [K, 2] int64 (y, x) kernel offsets; "dilations", mapping (device name, scene name, camera index, point size) at every point size above one to main's [H, W] float32 cpu dilation of its depth render at point size one without a mask, background set to positive infinity; "main_commit", the main commit that rendered them; and "scenes_digest", the hex sha256 of the "scenes.pt" bytes they were rendered from.
    """
    main_commit = subprocess.run(
        args=["git", "rev-parse", "HEAD"],
        cwd=main_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    scenes_digest = hashlib.sha256((output_dir / "scenes.pt").read_bytes()).hexdigest()
    main_branch_commit = subprocess.run(
        args=["git", "rev-parse", "main"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # The proof is against main as it stands, not an older checkout of it.
    assert main_commit == main_branch_commit, (
        "Expected the main checkout to sit at the commit this repo's main branch points at. "
        f"{main_repo=} {main_commit=} {main_branch_commit=}"
    )

    if (output_dir / "main_renders.pt").exists() and not force:
        cached = torch.load(output_dir / "main_renders.pt")
        if (
            cached["main_commit"] == main_commit
            and cached["scenes_digest"] == scenes_digest
        ):
            return cached
    subprocess.run(
        args=[
            sys.executable,
            str(Path(__file__).resolve().parent / "render_on_main.py"),
            "--scenes_path",
            str(output_dir / "scenes.pt"),
            "--output_path",
            str(output_dir / "main_renders.pt"),
        ],
        cwd=main_repo,
        env={
            **os.environ,
            "PYTHONPATH": str(main_repo),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        check=True,
    )
    main_renders = torch.load(output_dir / "main_renders.pt")
    main_renders.update(main_commit=main_commit, scenes_digest=scenes_digest)
    torch.save(main_renders, output_dir / "main_renders.pt")
    return main_renders


def compare_single_camera_to_main(
    scenes: List[Dict[str, Any]], main_renders: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compares every render this branch makes of one camera, handed over alone and, where the entry takes a batch, as a batch of one, with the one main made of the same scene, camera, renderer, point size and mask option.

    Args:
        scenes: The scene dicts build_scenes returns.
        main_renders: The dict load_or_render_on_main returns, whose "renders" maps (device name, scene name, camera index, renderer, point size, return_mask) to main's cpu render.

    Returns:
        One JSON-ready record per comparison: its "kind" ("camera", or "batch_of_one" for the depth entry handed a Cameras of one), "device", "scene", "camera" index, "renderer", "point_size" and "return_mask"; the "equal", "differing_elements", "nan_elements" and "max_abs_diff" compare_exactly returns against main's render; and "required", True at point size one on a scene other than "ties".
    """
    records = []
    for device in DEVICES:
        for scene in scenes:
            for camera_index in range(len(scene["cameras"])):
                for renderer in RENDERERS:
                    for point_size in POINT_SIZES:
                        for return_mask in RETURN_MASK_OPTIONS:
                            pc = build_point_cloud(scene=scene, device=device)
                            camera = build_camera(
                                scene=scene, camera_index=camera_index, device=device
                            )
                            output = render_single_camera(
                                renderer=renderer,
                                pc=pc,
                                camera=camera,
                                resolution=scene["resolution"],
                                return_mask=return_mask,
                                point_size=point_size,
                            )
                            main_output = main_renders["renders"][
                                (
                                    str(device),
                                    scene["name"],
                                    camera_index,
                                    renderer,
                                    point_size,
                                    return_mask,
                                )
                            ]
                            comparison = compare_exactly(
                                output=output, reference=main_output
                            )
                            records.append(
                                {
                                    "kind": "camera",
                                    "device": str(device),
                                    "scene": scene["name"],
                                    "camera": camera_index,
                                    "renderer": renderer,
                                    "point_size": point_size,
                                    "return_mask": return_mask,
                                    **comparison,
                                }
                            )

                            # --- The depth entry handed the same camera as a batch of one, its only slice against main's render
                            if renderer == "depth":
                                cameras = build_cameras(
                                    scene=scene,
                                    camera_indices=[camera_index],
                                    device=device,
                                )
                                batch_output = render_depth_from_point_cloud(
                                    pc=pc,
                                    camera=cameras,
                                    resolution=scene["resolution"],
                                    return_mask=return_mask,
                                    point_size=point_size,
                                )
                                if return_mask:
                                    # The map's and the mask's only slice.
                                    batch_slice = (
                                        batch_output[0][0],
                                        batch_output[1][0],
                                    )
                                else:
                                    batch_slice = batch_output[0]
                                comparison = compare_exactly(
                                    output=batch_slice, reference=main_output
                                )
                                records.append(
                                    {
                                        "kind": "batch_of_one",
                                        "device": str(device),
                                        "scene": scene["name"],
                                        "camera": camera_index,
                                        "renderer": renderer,
                                        "point_size": point_size,
                                        "return_mask": return_mask,
                                        **comparison,
                                    }
                                )

    # Above one pixel this branch's dilation grows a centred disc taking the nearest neighbour, and its depth entry applies it, where main did neither; at a tied depth this branch keeps the lowest point index, where main keeps the last point in order on cpu and the first on cuda.
    for record in records:
        record["required"] = record["point_size"] == 1.0 and record["scene"] != "ties"
    return records


def compare_batch_to_one_by_one(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compares, on this branch alone, one call over a scene's whole batch of cameras with each camera on its own, stage by stage, so the rounding CUDA's batched kernels introduce is told apart from a batching error.

    Args:
        scenes: The scene dicts build_scenes returns.

    Returns:
        One JSON-ready record per comparison of the batch's slice against the camera's own, each carrying its "kind", "device", "scene" and "camera" index: "prepare" records the point preparation with the "num_divide" both sides used (None, or 2 for four point chunks), carrying on cpu compare_exactly's "equal", "differing_elements", "nan_elements" and "max_abs_diff" over the kept rows and their point indices, and on cuda compare_preparations' "equal", "exact", "flipped_points" and "max_abs_diff"; "rasterize" records compare_exactly over the depth raster of one shared preparation, with its "return_mask"; "dilate" records compare_exactly over that raster's dilation, with its "point_size"; "depth" records compare_exactly over the depth entry's map and mask, with its "point_size" and "return_mask". Each also carries "required", True for all but a cuda "depth" record.
    """
    records = []
    for device in DEVICES:
        for scene in scenes:
            pc = build_point_cloud(scene=scene, device=device)
            cameras = build_cameras(
                scene=scene,
                camera_indices=list(range(len(scene["cameras"]))),
                device=device,
            )

            # --- The batch's point preparation against each camera's own, whole and then in chunks
            batch_preparations = {}
            # The second splits the points into chunks, so a chunk's own row count reaches the small-matrix kernels.
            for num_divide in (None, 2):
                batch_preparations[num_divide] = prepare_points_for_rendering(
                    pc=pc,
                    camera=cameras,
                    resolution=scene["resolution"],
                    num_divide=num_divide,
                )
                batch_points, batch_valid, _ = batch_preparations[num_divide]
                for camera_index in range(len(scene["cameras"])):
                    camera = build_camera(
                        scene=scene, camera_index=camera_index, device=device
                    )
                    points, _, original_data_indices = prepare_points_for_rendering(
                        pc=pc,
                        camera=camera,
                        resolution=scene["resolution"],
                        num_divide=num_divide,
                    )
                    if device.type == "cpu":
                        comparison = compare_exactly(
                            output=(
                                batch_points[camera_index][batch_valid[camera_index]],
                                batch_valid[camera_index].nonzero().squeeze(1),
                            ),
                            reference=(points, original_data_indices),
                        )
                    else:
                        # CUDA's batched inverse and product round unlike a single camera's.
                        comparison = compare_preparations(
                            output=(
                                batch_points[camera_index],
                                batch_valid[camera_index],
                            ),
                            reference=(points, original_data_indices),
                            pc=pc,
                            camera=camera,
                            resolution=scene["resolution"],
                        )
                    records.append(
                        {
                            "kind": "prepare",
                            "device": str(device),
                            "scene": scene["name"],
                            "camera": camera_index,
                            "num_divide": num_divide,
                            **comparison,
                        }
                    )

            # --- Rasterizing the batch against rasterizing each of its slices
            # One input handed to both sides, so the rasterizing stage is measured apart from the rounding before it.
            rendering_points, valid, _ = batch_preparations[None]
            for return_mask in RETURN_MASK_OPTIONS:
                batch_raster = render_depth_from_rendering_points(
                    rendering_points=rendering_points,
                    resolution=scene["resolution"],
                    ignore_value=float("inf"),
                    return_mask=return_mask,
                    valid=valid,
                )
                # The depth map of batch_raster.
                if return_mask:
                    batch_depth_map = batch_raster[0]
                else:
                    batch_depth_map = batch_raster
                for camera_index in range(len(scene["cameras"])):
                    slice_raster = render_depth_from_rendering_points(
                        rendering_points=rendering_points[camera_index],
                        resolution=scene["resolution"],
                        ignore_value=float("inf"),
                        return_mask=return_mask,
                        valid=valid[camera_index],
                    )
                    if return_mask:
                        # The map's and the mask's slice at camera_index.
                        batch_slice = (
                            batch_raster[0][camera_index],
                            batch_raster[1][camera_index],
                        )
                    else:
                        batch_slice = batch_raster[camera_index]
                    comparison = compare_exactly(
                        output=batch_slice, reference=slice_raster
                    )
                    records.append(
                        {
                            "kind": "rasterize",
                            "device": str(device),
                            "scene": scene["name"],
                            "camera": camera_index,
                            "return_mask": return_mask,
                            **comparison,
                        }
                    )

            # --- Dilating the batch's depth map against dilating each of its slices
            for point_size in POINT_SIZES:
                if point_size > 1.0:
                    batch_dilation = apply_point_size_postprocessing(
                        rendered_image=batch_depth_map,
                        depth_map=batch_depth_map,
                        point_size=point_size,
                        ignore_value=float("inf"),
                    )
                    for camera_index in range(len(scene["cameras"])):
                        slice_dilation = apply_point_size_postprocessing(
                            rendered_image=batch_depth_map[camera_index],
                            depth_map=batch_depth_map[camera_index],
                            point_size=point_size,
                            ignore_value=float("inf"),
                        )
                        comparison = compare_exactly(
                            output=batch_dilation[camera_index],
                            reference=slice_dilation,
                        )
                        records.append(
                            {
                                "kind": "dilate",
                                "device": str(device),
                                "scene": scene["name"],
                                "camera": camera_index,
                                "point_size": point_size,
                                **comparison,
                            }
                        )

            # --- The depth entry over the batch against each camera on its own
            for point_size in POINT_SIZES:
                for return_mask in RETURN_MASK_OPTIONS:
                    batch_output = render_depth_from_point_cloud(
                        pc=pc,
                        camera=cameras,
                        resolution=scene["resolution"],
                        return_mask=return_mask,
                        point_size=point_size,
                    )
                    for camera_index in range(len(scene["cameras"])):
                        camera = build_camera(
                            scene=scene, camera_index=camera_index, device=device
                        )
                        camera_output = render_depth_from_point_cloud(
                            pc=pc,
                            camera=camera,
                            resolution=scene["resolution"],
                            return_mask=return_mask,
                            point_size=point_size,
                        )
                        if return_mask:
                            # The map's and the mask's slice at camera_index.
                            batch_slice = (
                                batch_output[0][camera_index],
                                batch_output[1][camera_index],
                            )
                        else:
                            batch_slice = batch_output[camera_index]
                        comparison = compare_exactly(
                            output=batch_slice, reference=camera_output
                        )
                        records.append(
                            {
                                "kind": "depth",
                                "device": str(device),
                                "scene": scene["name"],
                                "camera": camera_index,
                                "point_size": point_size,
                                "return_mask": return_mask,
                                **comparison,
                            }
                        )

    # End to end, cuda carries the preparation's rounding into the render, which the "prepare", "rasterize" and "dilate" records account for between them.
    for record in records:
        record["required"] = record["kind"] != "depth" or record["device"] == "cpu"
    return records


def build_cameras(
    scene: Dict[str, Any], camera_indices: List[int], device: torch.device
) -> Cameras:
    """Builds the batch of the named cameras of a scene, the one input only this branch's renderers take.

    Args:
        scene: A scene dict build_scenes returns, whose "cameras" hold per camera its 0-dim float32 cpu "params" (fx / fy / cx / cy / h / w stated in scene["intr_convention"]) and its [4, 4] float32 cpu camera-to-world "extrinsics" in scene["extr_convention"].
        camera_indices: Indices into scene["cameras"] of the cameras to batch, in batch order.
        device: Device the batch is built on.

    Returns:
        The Cameras on device whose intrinsics params are each [B] and whose extrinsics are [B, 4, 4] in scene["extr_convention"], B being len(camera_indices).
    """
    params = {}
    for key in scene["cameras"][camera_indices[0]]["params"]:
        param_values = []
        for camera_index in camera_indices:
            param_values.append(scene["cameras"][camera_index]["params"][key])
        params[key] = torch.stack(param_values)
    intrinsics = build_camera_intrinsics(
        model=scene["model"],
        params=params,
        intr_convention=scene["intr_convention"],
        device=device,
    )
    extrinsics_list = []
    for camera_index in camera_indices:
        extrinsics_list.append(scene["cameras"][camera_index]["extrinsics"])
    extrinsics_matrices = torch.stack(extrinsics_list)
    extrinsics = CameraExtrinsics(
        extrinsics=extrinsics_matrices,
        extr_convention=scene["extr_convention"],
        device=device,
    )
    return Cameras(intrinsics=intrinsics, extrinsics=extrinsics, device=device)


def summarize_point_size_changes(main_renders: Dict[str, Any]) -> Dict[str, Any]:
    """Records the three facts that account for every difference from main above one pixel: which point sizes grow a different disc on each side, that main's depth-based entries ignore the point size altogether, and which neighbour each side's dilation keeps on one depth map.

    Args:
        main_renders: The dict load_or_render_on_main returns, whose "kernels" maps each point size to main's [K, 2] int64 (y, x) kernel offsets, whose "renders" maps (device name, scene name, camera index, renderer, point size, return_mask) to main's cpu render, and whose "dilations" maps (device name, scene name, camera index, point size) to main's dilation of its depth render at point size one without a mask, background set to positive infinity.

    Returns:
        The JSON-ready summary: per point size, whether this branch's kernel and main's hold the same set of (y, x) offsets; per depth-based renderer and point size, the tally {True: equal count, False: unequal count} of main's renders compare_exactly finds equal to main's render of the same device, scene, camera and mask option at point size one; and per point size above one, the same tally of this branch's dilations of main's depth render against main's dilation of the same device, scene, camera and point size.
    """
    summary = {}

    # --- Which point sizes grow a different disc on each side
    for point_size in POINT_SIZES:
        kernel_offsets = create_circular_kernel_offsets(
            point_size=point_size, device=torch.device("cpu")
        )
        summary[f"kernel offsets at point_size {point_size} same as main's"] = set(
            map(tuple, kernel_offsets.tolist())
        ) == set(map(tuple, main_renders["kernels"][point_size].tolist()))

    # --- Whether each of main's depth-based renders at every point size equals its render at point size one
    for (
        device_name,
        scene_name,
        camera_index,
        renderer,
        point_size,
        return_mask,
    ), render in main_renders["renders"].items():
        if renderer not in ("depth", "normal_2d"):
            continue
        comparison = compare_exactly(
            output=render,
            reference=main_renders["renders"][
                (device_name, scene_name, camera_index, renderer, 1.0, return_mask)
            ],
        )
        summary.setdefault(
            f"main {renderer} at point_size {point_size} equal to its point_size 1.0 render",
            {True: 0, False: 0},
        )[comparison["equal"]] += 1

    # --- Whether this branch's dilation of main's depth map equals main's dilation of it
    for (
        device_name,
        scene_name,
        camera_index,
        point_size,
    ), main_dilation in main_renders["dilations"].items():
        # The map render_on_main dilated with main's own dilation, main's depth entry filling the pixels no point lands on with its default ignore_value of -1.0.
        depth_map = main_renders["renders"][
            (device_name, scene_name, camera_index, "depth", 1.0, False)
        ].masked_fill(
            main_renders["renders"][
                (device_name, scene_name, camera_index, "depth", 1.0, False)
            ]
            == -1.0,
            float("inf"),
        )
        dilation = apply_point_size_postprocessing(
            rendered_image=depth_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float("inf"),
        )
        comparison = compare_exactly(output=dilation, reference=main_dilation)
        # main keeps the last nearer neighbour in kernel order, this branch the nearest.
        summary.setdefault(
            f"branch dilation of main's depth at point_size {point_size} equal to main's dilation",
            {True: 0, False: 0},
        )[comparison["equal"]] += 1
    return summary


def summarize_tie_changes(
    single_camera_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Records, per device and renderer, how many of the ties scene's point-size-one renders come out equal to main's, the one regime where main's choice between tied points depends on the device.

    Args:
        single_camera_records: The records compare_single_camera_to_main returns, each carrying its "device" name, "scene" name, "renderer", "point_size" and "equal".

    Returns:
        The JSON-ready summary: per device name and renderer, the tally {True: equal count, False: unequal count} of the ties scene's records at point size one, the "camera" and "batch_of_one" ones alike.
    """
    summary = {}
    for record in single_camera_records:
        if record["scene"] == "ties" and record["point_size"] == 1.0:
            summary.setdefault(
                f"ties on {record['device']}, {record['renderer']} at point_size 1.0 equal to main's",
                {True: 0, False: 0},
            )[record["equal"]] += 1
    return summary


def compare_preparations(
    output: Tuple[torch.Tensor, torch.Tensor],
    reference: Tuple[torch.Tensor, torch.Tensor],
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
) -> Dict[str, Any]:
    """Decides whether two preparations of one camera agree up to floating-point rounding, the test a cuda batch's preparation is held to.

    Args:
        output: The batch's preparation of the camera, on any device: its [N, 3] floating (x, y, depth) points, x and y in pixels and row i being point i of the cloud, and its [N] bool valid mask.
        reference: The camera's own preparation, on any device: its [M, 3] (x, y, depth) survivors in output's dtype and their [M] int64 original_data_indices into the cloud.
        pc: The PointCloud both preparations were made from, its xyz the [N, 3] world-space points.
        camera: The single Camera both preparations project through, its extrinsics camera-to-world in any pose convention and its intrinsics stated at their own resolution, which the preparation rescales to resolution before projecting.
        resolution: Render resolution as an (H, W) tuple, the image bounds both preparations cull against.

    Returns:
        A JSON-ready dict: "equal", True when every point both sides keep agrees in (x, y, depth) within tolerance and every point only one side keeps lies within tolerance of a cull boundary; "exact", whether compare_exactly finds the kept rows and their point indices identical; "flipped_points", how many points only one side keeps; and "max_abs_diff", compare_exactly's largest finite difference between the kept rows, None when the sides keep different counts.
    """

    def _validate_inputs() -> None:
        assert (
            output[0].ndim == 2
            and output[0].shape[1] == 3
            and output[0].is_floating_point()
        ), (
            "Expected the batch's points to be an [N, 3] floating tensor. "
            f"{output[0].shape=} {output[0].dtype=}"
        )
        assert (
            output[1].shape == output[0].shape[:1] and output[1].dtype == torch.bool
        ), (
            "Expected the batch's valid mask to be [N] bool over its points. "
            f"{output[1].shape=} {output[1].dtype=} {output[0].shape=}"
        )
        assert (
            reference[0].ndim == 2
            and reference[0].shape[1] == 3
            and reference[0].dtype == output[0].dtype
        ), (
            "Expected the camera's survivors to be [M, 3] in the batch's dtype. "
            f"{reference[0].shape=} {reference[0].dtype=} {output[0].dtype=}"
        )
        assert (
            reference[1].shape == reference[0].shape[:1]
            and reference[1].dtype == torch.int64
        ), (
            "Expected the camera's original_data_indices to be [M] int64 over its survivors. "
            f"{reference[1].shape=} {reference[1].dtype=} {reference[0].shape=}"
        )
        assert isinstance(pc, PointCloud), (
            "Expected pc to be a PointCloud. " f"{type(pc)=}"
        )
        assert isinstance(camera, Camera), (
            "Expected camera to be a single Camera. " f"{type(camera)=}"
        )

    _validate_inputs()

    points, valid = output[0].cpu(), output[1].cpu()
    # The single camera's survivors and the points they are.
    reference_points, reference_indices = reference[0].cpu(), reference[1].cpu()
    reference_valid = torch.zeros_like(valid).index_fill_(0, reference_indices, True)
    reference_rows = torch.zeros_like(points).index_copy_(
        0, reference_indices, reference_points
    )
    # The size of the numbers the world-to-camera transform rounds.
    magnitude = max(float(camera.extrinsics.center.norm()), float(pc.xyz.abs().max()))
    # Its focal lengths are the ones the preparation projects with at resolution.
    render_camera = camera.scale_intrinsics(resolution=resolution)
    camera_frame_tolerance = 4 * torch.finfo(points.dtype).eps * magnitude
    # Read per point off whichever side kept it.
    depth = torch.where(valid, points[:, 2], reference_rows[:, 2])
    # The projection multiplies camera-frame rounding by focal length over depth.
    tolerance = torch.stack(
        [
            camera_frame_tolerance * float(render_camera.intrinsics.fx) / depth,
            camera_frame_tolerance * float(render_camera.intrinsics.fy) / depth,
            torch.full_like(depth, camera_frame_tolerance),
        ],
        dim=1,
    )
    kept = valid & reference_valid
    # A point either side culls lands on no pixel, so its coordinates carry nothing to compare.
    points_close = bool(
        ((points[kept] - reference_rows[kept]).abs() <= tolerance[kept]).all()
    )
    flipped = valid != reference_valid
    # The side that culled it keeps no row, so which boundary it crossed is not on record.
    flipped_rows = torch.where(
        valid[flipped].unsqueeze(1), points[flipped], reference_rows[flipped]
    )
    flips_explained = bool(
        (
            torch.stack(
                [
                    flipped_rows[:, 0].abs(),
                    (flipped_rows[:, 0] - resolution[1]).abs(),
                    flipped_rows[:, 1].abs(),
                    (flipped_rows[:, 1] - resolution[0]).abs(),
                    flipped_rows[:, 2].abs(),
                ],
                dim=1,
            )
            <= tolerance[flipped][:, [0, 0, 1, 1, 2]]
        )
        .any(dim=1)
        .all()
    )
    # So the report shows how often rounding moved anything at all.
    exact = compare_exactly(
        output=(points[valid], valid.nonzero().squeeze(1)), reference=reference
    )
    return {
        "equal": points_close and flips_explained,
        "exact": exact["equal"],
        "flipped_points": int(flipped.sum()),
        "max_abs_diff": exact["max_abs_diff"],
    }


def compare_exactly(
    output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    reference: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Dict[str, Any]:
    """Decides whether two renders are the same result, which is the one test both equivalences are made of.

    Args:
        output: The render under test, on any device: one torch.Tensor (a lone map), or a tuple of torch.Tensors such as a (map, [H, W] bool mask) pair or a ([N, 3] float32 points, [N] bool valid) pair.
        reference: The render output is held against, in the same form and on any device.

    Returns:
        A JSON-ready dict: "equal", True when both hold as many tensors and each pair agrees in shape, in dtype and at every element, a NaN against a NaN counting as agreement; "differing_elements", per pair the count of elements that disagree that way, None for a pair of two shapes; "nan_elements", per pair the count of positions where both hold NaN, None for a pair of two shapes; and "max_abs_diff", the largest absolute difference over the positions both floating members of a pair of one shape hold finite, None when no such pair exists. Both per-pair lists are empty when the tensor counts differ.
    """

    def _validate_inputs() -> None:
        for render in (output, reference):
            assert isinstance(render, torch.Tensor) or (
                isinstance(render, tuple)
                and all(isinstance(member, torch.Tensor) for member in render)
            ), (
                "Expected each render to be a torch.Tensor or a tuple of torch.Tensors. "
                f"{type(render)=}"
            )

    _validate_inputs()

    def _normalize_inputs(
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        reference: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        # A lone map becoming a tuple of one.
        if not isinstance(output, tuple):
            output = (output,)
        if not isinstance(reference, tuple):
            reference = (reference,)
        cpu_members = []
        for member in output:
            cpu_members.append(member.cpu())
        output = tuple(cpu_members)
        cpu_reference_members = []
        for member in reference:
            cpu_reference_members.append(member.cpu())
        reference = tuple(cpu_reference_members)
        return output, reference

    output, reference = _normalize_inputs(output=output, reference=reference)

    # --- Pair the tensors up and mark, per pair of one shape, the elements that disagree, a NaN against a NaN agreeing
    if len(output) == len(reference):
        pairs = list(zip(output, reference, strict=True))
    else:
        pairs = []
    disagreements = []
    for member, reference_member in pairs:
        if member.shape == reference_member.shape:
            disagreements.append(
                (member != reference_member)
                & ~(torch.isnan(member) & torch.isnan(reference_member))
            )
        else:
            disagreements.append(None)

    # --- The verdict and the counts behind it
    equal = len(output) == len(reference)
    for (member, reference_member), disagreement in zip(
        pairs, disagreements, strict=True
    ):
        equal = (
            equal
            and member.shape == reference_member.shape
            and member.dtype == reference_member.dtype
            and not bool(disagreement.any())
        )
    # None for a pair of two shapes.
    differing_elements = []
    for disagreement in disagreements:
        if disagreement is None:
            differing_elements.append(None)
        else:
            differing_elements.append(int(disagreement.sum()))
    # None for a pair of two shapes.
    nan_elements = []
    for member, reference_member in pairs:
        if member.shape == reference_member.shape:
            nan_elements.append(
                int((torch.isnan(member) & torch.isnan(reference_member)).sum())
            )
        else:
            nan_elements.append(None)
    # None when no pair is of one shape with both members floating.
    max_abs_diff = None
    for member, reference_member in pairs:
        if (
            member.shape == reference_member.shape
            and member.is_floating_point()
            and reference_member.is_floating_point()
        ):
            # Every difference is non-negative, so a None before the first such pair counts as 0.0.
            max_abs_diff = max(
                max_abs_diff or 0.0,
                float(
                    torch.where(
                        torch.isfinite(member) & torch.isfinite(reference_member),
                        (member - reference_member).abs(),
                        0.0,
                    ).max()
                ),
            )
    return {
        "equal": equal,
        "differing_elements": differing_elements,
        "nan_elements": nan_elements,
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    main()
