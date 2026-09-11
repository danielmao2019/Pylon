import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
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

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import (
    create_circular_kernel_offsets,
)
from models.three_d.point_cloud.render.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud


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

    # --- Report: the commits, the devices, both record lists with their required checks tallied, and the point-size summary
    dod_1_required = [record for record in single_camera_records if record["required"]]
    above_one_pixel = {}
    for kind, renderer, point_size in sorted(
        {
            (record["kind"], record["renderer"], record["point_size"])
            for record in single_camera_records
            if not record["required"]
        }
    ):
        group = [
            record
            for record in single_camera_records
            if (record["kind"], record["renderer"], record["point_size"])
            == (kind, renderer, point_size)
        ]
        above_one_pixel[f"{kind} {renderer} point_size {point_size}"] = {
            "total": len(group),
            "equal": sum(record["equal"] for record in group),
            "differing_elements": sum(
                count
                for record in group
                for count in record["differing_elements"]
                if count is not None
            ),
        }
    report = {
        "main_commit": main_renders["main_commit"],
        "branch_commit": subprocess.run(
            args=["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "branch_worktree_clean": subprocess.run(
            args=["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        == "",
        "devices": [str(device) for device in DEVICES],
        "dod_1": {
            "required": {
                "total": len(dod_1_required),
                "equal": sum(record["equal"] for record in dod_1_required),
                "failures": [
                    record for record in dod_1_required if not record["equal"]
                ],
            },
            "above_one_pixel": above_one_pixel,
            "records": single_camera_records,
        },
        "dod_2": {
            "required": {
                "total": len(batch_records),
                "equal": sum(record["equal"] for record in batch_records),
                "failures": [record for record in batch_records if not record["equal"]],
            },
            "records": batch_records,
        },
        "point_size_summary": point_size_summary,
    }
    (output_dir / "equivalence_report.json").write_text(json.dumps(report, indent=2))

    # --- One summary line per tally
    print(
        f"commits: main {report['main_commit']}, branch {report['branch_commit']}, "
        f"branch worktree clean {report['branch_worktree_clean']}, devices {report['devices']}"
    )
    print(
        f"dod_1 required (point_size 1.0): {report['dod_1']['required']['equal']}/"
        f"{report['dod_1']['required']['total']} equal, "
        f"{len(report['dod_1']['required']['failures'])} failures"
    )
    for label, tally in above_one_pixel.items():
        print(
            f"dod_1 above one pixel, {label}: {tally['equal']}/{tally['total']} equal, "
            f"{tally['differing_elements']} differing elements"
        )
    print(
        f"dod_2 required: {report['dod_2']['required']['equal']}/"
        f"{report['dod_2']['required']['total']} equal, "
        f"{len(report['dod_2']['required']['failures'])} failures"
    )
    for label, entry in point_size_summary.items():
        print(f"point-size summary, {label}: {entry}")

    if any(
        record["required"] and not record["equal"]
        for record in single_camera_records + batch_records
    ):
        raise SystemExit(1)


def load_or_build_scenes(output_dir: Path, force: bool) -> List[Dict[str, Any]]:
    """Returns the fixed seeded scenes both checkouts render, from output_dir / "scenes.pt" unless it is missing or force asks for a rebuild.

    Args:
        output_dir: This task's outputs/ directory.
        force: Whether to rebuild the scenes even when "scenes.pt" exists.

    Returns:
        The scene dicts build_scenes returns.
    """
    scenes_path = output_dir / "scenes.pt"
    if scenes_path.exists() and not force:
        return torch.load(scenes_path)
    scenes = build_scenes()
    torch.save(scenes, scenes_path)
    return scenes


def build_scenes() -> List[Dict[str, Any]]:
    """Builds scenes that reach every regime the batching changes: many points per pixel, points culled behind or beside some cameras but not others, a render resolution the intrinsics are rescaled to, and all three pose conventions.

    Args:
        None.

    Returns:
        The scene dicts, every tensor on cpu: "name"; the cloud as "xyz" [N, 3] float32 world-space points, "rgb" [N, 3] colours (float32 in [0, 1], or uint8 in [0, 255]), "labels" [N] int64 labels below twenty and "normals" [N, 3] float32 world-space unit normals; the intrinsics "model" ("pinhole") and "intr_convention" ("standard"); the pose "extr_convention" ("opengl", "opencv" or "standard"); the render "resolution" as (H, W); and "cameras", a list of {"params": 0-dim float32 fx / fy / cx / cy / h / w, "extrinsics": [4, 4] float32 camera-to-world in the scene's extr_convention}.
    """
    generator = torch.Generator().manual_seed(0)

    # --- The three scenes: a seeded cloud about the origin, how many cameras sit on a sphere of what radius around it, their base focal length, and the resolution the intrinsics state versus the one rendered
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

    # The fixed axis change per pose convention: an opencv camera-to-world rotation right-multiplied by it is the same pose in that convention.
    axis_changes = {
        "opencv": torch.eye(3),
        "opengl": torch.diag(torch.tensor([1.0, -1.0, -1.0])),
        "standard": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    }
    for scene in (collisions, culling, sparse):
        # --- Per-point colours, labels and normals
        num_points = scene["xyz"].shape[0]
        rgb_dtype = scene.pop("rgb_dtype")
        scene["rgb"] = (
            torch.randint(
                low=0,
                high=256,
                size=(num_points, 3),
                generator=generator,
                dtype=torch.uint8,
            )
            if rgb_dtype == torch.uint8
            else torch.rand(num_points, 3, generator=generator)
        )
        scene["labels"] = torch.randint(
            low=0,
            high=20,
            size=(num_points,),
            generator=generator,
            dtype=torch.int64,
        )
        scene["normals"] = torch.nn.functional.normalize(
            torch.randn(num_points, 3, generator=generator), dim=-1
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
        focal = scene.pop("focal")
        stated_height, stated_width = scene.pop("stated_resolution")
        scene["cameras"] = [
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
            for camera_index in range(num_cameras)
        ]
    return [collisions, culling, sparse]


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
    intrinsics = build_camera_intrinsics(
        model=scene["model"],
        params={
            key: torch.stack(
                [
                    scene["cameras"][camera_index]["params"][key]
                    for camera_index in camera_indices
                ]
            )
            for key in scene["cameras"][camera_indices[0]]["params"]
        },
        intr_convention=scene["intr_convention"],
        device=device,
    )
    extrinsics = CameraExtrinsics(
        extrinsics=torch.stack(
            [
                scene["cameras"][camera_index]["extrinsics"]
                for camera_index in camera_indices
            ]
        ),
        extr_convention=scene["extr_convention"],
        device=device,
    )
    return Cameras(intrinsics=intrinsics, extrinsics=extrinsics, device=device)


def load_or_render_on_main(
    main_repo: Path, output_dir: Path, force: bool
) -> Dict[str, Any]:
    """Returns main's renders of the scenes, from output_dir / "main_renders.pt" unless it is missing, was rendered at another main commit, or force asks for a rerender.

    Args:
        main_repo: Absolute path of a checkout of this repo's main branch.
        output_dir: This task's outputs/ directory, already holding "scenes.pt".
        force: Whether to rerender even when "main_renders.pt" exists at main's commit.

    Returns:
        The dict of "renders", mapping (device name, scene name, camera index, renderer, point size, return_mask) to main's cpu render (the map, or the (map, [H, W] bool mask) tuple when return_mask is True); "kernels", mapping each point size to main's [K, 2] int64 (y, x) kernel offsets; and "main_commit", the main commit that rendered them.
    """
    main_commit = subprocess.run(
        args=["git", "rev-parse", "HEAD"],
        cwd=main_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
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

    renders_path = output_dir / "main_renders.pt"
    if renders_path.exists() and not force:
        cached = torch.load(renders_path)
        if cached["main_commit"] == main_commit:
            return cached
    subprocess.run(
        args=[
            sys.executable,
            str(Path(__file__).resolve().parent / "render_on_main.py"),
            "--scenes_path",
            str(output_dir / "scenes.pt"),
            "--output_path",
            str(renders_path),
        ],
        cwd=main_repo,
        env={
            **os.environ,
            "PYTHONPATH": str(main_repo),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        check=True,
    )
    main_renders = torch.load(renders_path)
    main_renders["main_commit"] = main_commit
    torch.save(main_renders, renders_path)
    return main_renders


def compare_single_camera_to_main(
    scenes: List[Dict[str, Any]], main_renders: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compares every render this branch makes of one camera, handed over alone and, where the entry takes a batch, as a batch of one, with the one main made of the same scene, camera, renderer, point size and mask option.

    Args:
        scenes: The scene dicts build_scenes returns.
        main_renders: The dict load_or_render_on_main returns, whose "renders" maps (device name, scene name, camera index, renderer, point size, return_mask) to main's cpu render.

    Returns:
        One JSON-ready record per comparison: its "kind" ("camera", or "batch_of_one" for the depth entry handed a Cameras of one), "device", "scene", "camera" index, "renderer", "point_size" and "return_mask"; the "equal", "differing_elements", "nan_elements" and "max_abs_diff" compare_exactly returns against main's render; and "required", True at point size one.
    """
    records = []
    for device, scene, camera_index, renderer, point_size, return_mask in (
        (device, scene, camera_index, renderer, point_size, return_mask)
        for device in DEVICES
        for scene in scenes
        for camera_index in range(len(scene["cameras"]))
        for renderer in RENDERERS
        for point_size in POINT_SIZES
        for return_mask in RETURN_MASK_OPTIONS
    ):
        pc = build_point_cloud(scene=scene, device=device)
        camera = build_camera(scene=scene, camera_index=camera_index, device=device)
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
        comparison = compare_exactly(output=output, reference=main_output)
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
                scene=scene, camera_indices=[camera_index], device=device
            )
            batch_output = render_depth_from_point_cloud(
                pc=pc,
                camera=cameras,
                resolution=scene["resolution"],
                return_mask=return_mask,
                point_size=point_size,
            )
            comparison = compare_exactly(
                output=(
                    tuple(member[0] for member in batch_output)
                    if return_mask
                    else batch_output[0]
                ),
                reference=main_output,
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

    # Above one pixel this branch's dilation grows a centred disc taking the nearest neighbour, and its depth entry applies it, where main did neither.
    for record in records:
        record["required"] = record["point_size"] == 1.0
    return records


def compare_batch_to_one_by_one(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compares, on this branch alone, what one call over a scene's whole batch of cameras returns with what each camera returns on its own.

    Args:
        scenes: The scene dicts build_scenes returns.

    Returns:
        One JSON-ready record per comparison: its "kind" ("prepare" for the prepared points and valid mask, "depth" for the depth entry's map and mask), "device", "scene", "camera" index, "renderer", "point_size" and "return_mask" (None for "prepare"); the "equal", "differing_elements", "nan_elements" and "max_abs_diff" compare_exactly returns for the batch's slice against the camera's own; and "required", always True.
    """
    records = []
    for device, scene in ((device, scene) for device in DEVICES for scene in scenes):
        pc = build_point_cloud(scene=scene, device=device)
        cameras = build_cameras(
            scene=scene,
            camera_indices=list(range(len(scene["cameras"]))),
            device=device,
        )
        batch_points, batch_valid = prepare_points_for_rendering(
            pc=pc, camera=cameras, resolution=scene["resolution"]
        )

        # --- The prepared points and valid mask of the batch against each camera's own
        for camera_index in range(len(scene["cameras"])):
            camera = build_camera(scene=scene, camera_index=camera_index, device=device)
            points, valid = prepare_points_for_rendering(
                pc=pc, camera=camera, resolution=scene["resolution"]
            )
            comparison = compare_exactly(
                output=(batch_points[camera_index], batch_valid[camera_index]),
                reference=(points, valid),
            )
            records.append(
                {
                    "kind": "prepare",
                    "device": str(device),
                    "scene": scene["name"],
                    "camera": camera_index,
                    "renderer": None,
                    "point_size": None,
                    "return_mask": None,
                    **comparison,
                }
            )

        # --- The depth entry over the batch against each camera on its own
        for point_size, return_mask in (
            (point_size, return_mask)
            for point_size in POINT_SIZES
            for return_mask in RETURN_MASK_OPTIONS
        ):
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
                comparison = compare_exactly(
                    output=(
                        tuple(member[camera_index] for member in batch_output)
                        if return_mask
                        else batch_output[camera_index]
                    ),
                    reference=camera_output,
                )
                records.append(
                    {
                        "kind": "depth",
                        "device": str(device),
                        "scene": scene["name"],
                        "camera": camera_index,
                        "renderer": "depth",
                        "point_size": point_size,
                        "return_mask": return_mask,
                        **comparison,
                    }
                )

    for record in records:
        record["required"] = True
    return records


def summarize_point_size_changes(main_renders: Dict[str, Any]) -> Dict[str, Any]:
    """Records the two facts that account for every difference from main above one pixel: which point sizes grow a different disc on each side, and that main's depth-based entries ignore the point size altogether.

    Args:
        main_renders: The dict load_or_render_on_main returns, whose "kernels" maps each point size to main's [K, 2] int64 (y, x) kernel offsets and whose "renders" maps (device name, scene name, camera index, renderer, point size, return_mask) to main's cpu render.

    Returns:
        The JSON-ready summary: per point size, whether this branch's kernel and main's hold the same set of (y, x) offsets, with each side's offset count; and per depth-based renderer and point size, how many of main's renders compare_exactly finds equal to main's render of the same device, scene, camera and mask option at point size one.
    """
    summary = {}

    # --- Which point sizes grow a different disc on each side
    for point_size in POINT_SIZES:
        kernel_offsets = create_circular_kernel_offsets(
            point_size=point_size, device=torch.device("cpu")
        )
        summary[f"kernel offsets at point_size {point_size}"] = {
            "same_offsets": {tuple(offset) for offset in kernel_offsets.tolist()}
            == {
                tuple(offset) for offset in main_renders["kernels"][point_size].tolist()
            },
            "branch_count": kernel_offsets.shape[0],
            "main_count": main_renders["kernels"][point_size].shape[0],
        }

    # --- Whether each of main's depth-based renders at every point size equals its render at point size one
    depth_based_renderers = ("depth", "normal_2d")
    summary.update(
        {
            f"main {renderer} at point_size {point_size} equal to its point_size 1.0 render": {
                "total": 0,
                "equal": 0,
            }
            for renderer in depth_based_renderers
            for point_size in POINT_SIZES
        }
    )
    for (
        device_name,
        scene_name,
        camera_index,
        renderer,
        point_size,
        return_mask,
    ), render in main_renders["renders"].items():
        if renderer not in depth_based_renderers:
            continue
        comparison = compare_exactly(
            output=render,
            reference=main_renders["renders"][
                (device_name, scene_name, camera_index, renderer, 1.0, return_mask)
            ],
        )
        tally = summary[
            f"main {renderer} at point_size {point_size} equal to its point_size 1.0 render"
        ]
        tally["total"] += 1
        tally["equal"] += comparison["equal"]
    return summary


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
        output = tuple(
            member.cpu()
            for member in (output if isinstance(output, tuple) else (output,))
        )
        reference = tuple(
            member.cpu()
            for member in (reference if isinstance(reference, tuple) else (reference,))
        )
        return output, reference

    output, reference = _normalize_inputs(output=output, reference=reference)

    # --- Pair the tensors up and mark, per pair of one shape, the elements that disagree, a NaN against a NaN agreeing
    pairs = (
        list(zip(output, reference, strict=True))
        if len(output) == len(reference)
        else []
    )
    disagreements = [
        (
            (member != reference_member)
            & ~(torch.isnan(member) & torch.isnan(reference_member))
            if member.shape == reference_member.shape
            else None
        )
        for member, reference_member in pairs
    ]

    # --- The verdict and the counts behind it
    equal = len(output) == len(reference) and all(
        member.shape == reference_member.shape
        and member.dtype == reference_member.dtype
        and not bool(disagreement.any())
        for (member, reference_member), disagreement in zip(
            pairs, disagreements, strict=True
        )
    )
    differing_elements = [
        None if disagreement is None else int(disagreement.sum())
        for disagreement in disagreements
    ]
    nan_elements = [
        (
            int((torch.isnan(member) & torch.isnan(reference_member)).sum())
            if member.shape == reference_member.shape
            else None
        )
        for member, reference_member in pairs
    ]
    max_abs_diff = max(
        (
            float(
                torch.where(
                    torch.isfinite(member) & torch.isfinite(reference_member),
                    (member - reference_member).abs(),
                    0.0,
                ).max()
            )
            for member, reference_member in pairs
            if member.shape == reference_member.shape
            and member.is_floating_point()
            and reference_member.is_floating_point()
        ),
        default=None,
    )
    return {
        "equal": equal,
        "differing_elements": differing_elements,
        "nan_elements": nan_elements,
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    main()
