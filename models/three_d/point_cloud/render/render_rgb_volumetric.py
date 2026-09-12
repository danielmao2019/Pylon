import itertools
import logging
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select
from models.three_d.point_cloud.render.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from models.three_d.point_cloud.render.render_rgb import (
    render_rgb_from_point_cloud,
)
from models.three_d.splatfacto.load_splatfacto import load_splatfacto_model
from models.three_d.splatfacto.render import render_rgb_from_splatfacto


def render_rgb_from_point_cloud_volumetric(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    debug: bool = False,
) -> torch.Tensor:
    """Render one view volumetrically: cull to the points that project, ring the view with auxiliary cameras, train a splatfacto model on that tiny dataset, and evaluate it back at the original camera.

    Args:
        pc: PointCloud with xyz and rgb fields in world coordinates.
        camera: Single Camera to render through; its native image size is read as twice its principal point.
        resolution: Target resolution as an (H, W) tuple, the native size divided by one of the downscale factors 1, 2, 4, 8.
        debug: If True, keep the dataset workspace at ./test_volumetric_rendering instead of a temporary directory deleted afterwards.

    Returns:
        [3, H, W] float32 torch.Tensor RGB image in [0, 1] rendered by the trained splatfacto model at camera, on the device of pc.
    """
    total_start = time.time()
    logging.info("[volumetric] Pipeline start")

    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"

    render_height, render_width = resolution
    assert render_height > 0 and render_width > 0, "Render resolution must be positive"

    intrinsics = camera.intrinsics
    extrinsics = camera.extrinsics
    convention = camera.extrinsics.extr_convention

    native_width = int(round(float(intrinsics.cx * 2.0)))
    native_height = int(round(float(intrinsics.cy * 2.0)))
    assert (
        native_width > 0 and native_height > 0
    ), "Invalid image dimensions inferred from intrinsics"
    downscale_ratio_w = native_width / float(render_width)
    downscale_ratio_h = native_height / float(render_height)
    downscale_estimate = 0.5 * (downscale_ratio_w + downscale_ratio_h)

    valid_factors = [1, 2, 4, 8]
    downscale_factor = min(
        valid_factors, key=lambda factor: abs(downscale_estimate - factor)
    )

    assert (
        math.isfinite(downscale_estimate)
        and abs(downscale_ratio_w - downscale_factor) <= 0.01
        and abs(downscale_ratio_h - downscale_factor) <= 0.01
    ), (
        "Render resolution does not correspond to a supported downscale factor. "
        f"Expected close to one of {valid_factors}, got ratios "
        f"({downscale_ratio_w:.3f}, {downscale_ratio_h:.3f}) with abs errors "
        f"({abs(downscale_ratio_w - downscale_factor):.3e}, {abs(downscale_ratio_h - downscale_factor):.3e})."
    )

    stage_start = time.time()
    _, _, image_plane_points_indices = prepare_points_for_rendering(
        pc=pc,
        camera=camera,
        resolution=resolution,
    )
    pc = Select(indices=image_plane_points_indices)(pc)
    aux_cameras = gen_auxiliary_cameras(
        points=pc.xyz,
        camera=camera,
    )
    train_extrinsics = [extrinsics] + [
        aux_camera.extrinsics for aux_camera in aux_cameras
    ]
    logging.info(
        "[volumetric] Visibility filtering & auxiliary cameras done in %.2fs (cameras=%d)",
        time.time() - stage_start,
        len(train_extrinsics),
    )

    stage_start = time.time()
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for _extrinsics in train_extrinsics:
        render_camera = Camera(
            intrinsics=intrinsics,
            extrinsics=_extrinsics,
            device=pc.device,
        )
        image, mask = render_rgb_from_point_cloud(
            pc=pc,
            camera=render_camera,
            resolution=resolution,
            return_mask=True,
        )
        images.append(image)
        masks.append(mask)
    logging.info(
        "[volumetric] Rendered %d base RGB/mask pairs in %.2fs",
        len(images),
        time.time() - stage_start,
    )

    target_device = pc.xyz.device

    if debug:
        tempdir = Path("./test_volumetric_rendering")
        tempdir.mkdir(parents=True, exist_ok=True)
        cleanup_fn = None
        logging.info("[volumetric] Workspace retained at %s", tempdir)
    else:
        temp_dir_context = tempfile.TemporaryDirectory()
        tempdir = Path(temp_dir_context.name)
        cleanup_fn = temp_dir_context.cleanup

    try:
        stage_start = time.time()
        logging.info(
            "[volumetric] Writing images, masks, and point cloud to %s", tempdir
        )
        _create_images(
            images=images,
            output_root=tempdir,
            downscale_factor=downscale_factor,
        )
        _create_masks(
            masks=masks,
            output_root=tempdir,
            downscale_factor=downscale_factor,
        )
        _create_ply(pc=pc, output_root=tempdir)
        _create_nerfstudio(
            intrinsics=intrinsics,
            train_extrinsics=train_extrinsics,
            eval_extrinsics=extrinsics,
            convention=convention,
            output_root=tempdir,
        )
        logging.info(
            "[volumetric] Dataset artifacts written in %.2fs",
            time.time() - stage_start,
        )

        dataset_root = Path(tempdir)
        stage_start = time.time()
        model_dir = _run_ns_train_splatfacto(
            dataset_root=dataset_root,
            downscale_factor=downscale_factor,
        )
        logging.info(
            "[volumetric] ns-train completed in %.2fs",
            time.time() - stage_start,
        )

        stage_start = time.time()
        _assert_checkpoint_exists(model_dir=model_dir)
        pipeline = load_splatfacto_model(model_dir=str(model_dir), device=target_device)
        logging.info(
            "[volumetric] Loaded trained model in %.2fs",
            time.time() - stage_start,
        )

        stage_start = time.time()
        rendered_image = render_rgb_from_splatfacto(
            model=pipeline,
            camera=camera,
            resolution=resolution,
        )
        logging.info(
            "[volumetric] Evaluation render finished in %.2fs",
            time.time() - stage_start,
        )
    finally:
        if cleanup_fn is not None:
            cleanup_fn()

    logging.info(
        "[volumetric] Pipeline finished in %.2fs",
        time.time() - total_start,
    )
    return rendered_image.to(device=target_device)


def gen_auxiliary_cameras(
    points: torch.Tensor,
    camera: Camera,
) -> List[Camera]:
    """Ring the primary view with offset cameras, so one input view still gives a volumetric fit a spread of poses to train against.

    Args:
        points: [N, 3] float torch.Tensor of world-space points whose mean is the centre the primary camera's distance is measured against.
        camera: Primary Camera the auxiliary cameras are stepped away from.

    Returns:
        List of 26 Camera instances, one per non-zero direction in {-1, 0, 1}^3, each carrying the primary camera's intrinsics and rotation with its position stepped half the camera-to-centre distance along that unit direction, in the standard extrinsics convention.
    """
    device = points.device

    center = points.mean(dim=0).to(device=device, dtype=torch.float32)
    camera = camera.to(device=device, extr_convention="standard")
    extrinsics_standard = camera.extrinsics.extrinsics

    camera_position = extrinsics_standard[:3, 3]
    distance = torch.linalg.norm(camera_position - center)
    assert distance.item() > 0, "Primary camera coincides with point cloud center"
    step = distance * 0.5

    direction_specs = [
        (
            direction := torch.tensor(
                (float(x), float(y), float(z)),
                device=device,
                dtype=torch.float32,
            )
        )
        / torch.linalg.norm(direction)
        for x, y, z in itertools.product([-1, 0, 1], repeat=3)
        if not (x == 0 and y == 0 and z == 0)
    ]

    auxiliary_cameras: List[Camera] = []

    for direction_unit in direction_specs:
        assert direction_unit.shape == (3,), "Auxiliary camera direction must be 3D"

        position = camera_position + direction_unit * step
        aux_standard = torch.zeros((4, 4), device=device, dtype=torch.float32)
        aux_standard[3, 3] = 1.0
        aux_standard[:3, :3] = extrinsics_standard[:3, :3]
        aux_standard[:3, 3] = position

        aux_extrinsics = CameraExtrinsics(
            extrinsics=aux_standard,
            extr_convention="standard",
            device=device,
        )
        aux_camera = Camera(
            intrinsics=camera.intrinsics,
            extrinsics=aux_extrinsics,
            device=device,
        ).to(extr_convention=camera.extrinsics.extr_convention)
        auxiliary_cameras.append(aux_camera)

    return auxiliary_cameras


def _create_images(
    images: List[torch.Tensor],
    output_root: str,
    downscale_factor: int,
) -> None:
    """Write the rendered RGB tensors out as the downscale-suffixed images directory a nerfstudio dataset reads.

    Args:
        images: List of [3, H, W] float torch.Tensor RGB images with values in [0, 1], one per training camera.
        output_root: Dataset root directory the images directory is created under.
        downscale_factor: Downscale factor the images were rendered at; above 1 it suffixes the directory name as images_{downscale_factor}.

    Returns:
        None.
    """
    root = Path(output_root)
    suffix = f"_{downscale_factor}" if downscale_factor > 1 else ""
    image_dir = root / f"images{suffix}"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images):
        tensor = image.detach().cpu().clamp(0.0, 1.0).float()
        array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        file_path = image_dir / f"image_{idx:02d}.png"
        Image.fromarray(array).save(file_path)


def _create_masks(
    masks: List[torch.Tensor],
    output_root: str,
    downscale_factor: int,
) -> None:
    """Write the rendered coverage masks out as the downscale-suffixed masks directory a nerfstudio dataset reads.

    Args:
        masks: List of [H, W] bool torch.Tensor coverage masks, True where a point was rendered, one per training camera.
        output_root: Dataset root directory the masks directory is created under.
        downscale_factor: Downscale factor the masks were rendered at; above 1 it suffixes the directory name as masks_{downscale_factor}.

    Returns:
        None.
    """
    root = Path(output_root)
    suffix = f"_{downscale_factor}" if downscale_factor > 1 else ""
    mask_dir = root / f"masks{suffix}"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, mask in enumerate(masks):
        tensor = mask.detach().cpu().bool()
        array = tensor.numpy().astype(np.uint8) * 255
        file_path = mask_dir / f"mask_{idx:02d}.png"
        Image.fromarray(array, mode="L").save(file_path)


def _create_ply(pc: PointCloud, output_root: str) -> None:
    """Write the culled point cloud as the point_cloud.ply the nerfstudio dataset seeds its gaussians from.

    Args:
        pc: PointCloud of the points that project into the primary view.
        output_root: Dataset root directory the point_cloud.ply file is written into.

    Returns:
        None.
    """
    root = Path(output_root)
    ply_path = root / "point_cloud.ply"
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    save_point_cloud(pc, str(ply_path))


def _create_nerfstudio(cameras: List[Camera], output_root: Path) -> None:
    """Write the transforms.json a nerfstudio dataset is read through, carrying the shared intrinsics beside every training pose.

    Args:
        cameras: Non-empty list of named Camera instances, one per training pose, whose intrinsics the first camera's intrinsics stand for.
        output_root: Dataset root directory the transforms.json file is written into.

    Returns:
        None.
    """
    root = Path(output_root)
    assert cameras, "At least one camera required to write transforms.json"
    nerfstudio_path = root / "transforms.json"
    nerfstudio_path.parent.mkdir(parents=True, exist_ok=True)

    camera_names = [camera.name for camera in cameras]
    assert all(name is not None for name in camera_names), f"{camera_names=}"

    camera_intrinsics = cameras[0].intrinsics
    intrinsic_params = {
        "fl_x": float(camera_intrinsics.fx),
        "fl_y": float(camera_intrinsics.fy),
        "cx": float(camera_intrinsics.cx),
        "cy": float(camera_intrinsics.cy),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }
    resolution = (
        int(round(float(camera_intrinsics.cy * 2.0))),
        int(round(float(camera_intrinsics.cx * 2.0))),
    )
    camera_model = "OPENCV"
    intrinsics = torch.tensor(
        [
            [camera_intrinsics.fx, 0.0, camera_intrinsics.cx],
            [0.0, camera_intrinsics.fy, camera_intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=cameras[0].device,
    )
    applied_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    batched_intrinsics = build_camera_intrinsics(
        model=camera_intrinsics.model,
        params={
            key: torch.stack(
                [camera.intrinsics.params[key] for camera in cameras], dim=0
            )
            for key in camera_intrinsics.params
        },
        intr_convention=camera_intrinsics.intr_convention,
    )
    batched_extrinsics = CameraExtrinsics(
        extrinsics=torch.stack(
            [camera.extrinsics.extrinsics for camera in cameras], dim=0
        ),
        extr_convention=cameras[0].extrinsics.extr_convention,
    )
    nerfstudio_cameras = Cameras(
        intrinsics=batched_intrinsics,
        extrinsics=batched_extrinsics,
        names=camera_names,
        ids=[camera.id for camera in cameras],
        device=cameras[0].device,
    )
    modalities = ["image"]
    payload: Dict[str, Any] = {}
    nerfstudio_data = NerfStudio_Data(
        data=payload,
        device=cameras[0].device,
        intrinsic_params=intrinsic_params,
        resolution=resolution,
        camera_model=camera_model,
        intrinsics=intrinsics,
        applied_transform=applied_transform,
        ply_file_path="point_cloud.ply",
        cameras=nerfstudio_cameras,
        modalities=modalities,
        train_filenames=None,
        val_filenames=None,
        test_filenames=None,
    )
    nerfstudio_data.save(output_path=nerfstudio_path)


def _run_ns_train_splatfacto(
    dataset_root: Path,
    downscale_factor: int,
) -> Path:
    """Train a splatfacto model on the written dataset by shelling out to nerfstudio's ns-train, handing back the run directory it produced.

    Args:
        dataset_root: Root directory of the nerfstudio dataset; ns-train writes its runs under its outputs subdirectory.
        downscale_factor: Downscale factor of the images the dataset carries, forwarded to ns-train.

    Returns:
        Path of the run directory holding the newest config.yml under dataset_root / "outputs".
    """
    output_dir = dataset_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ns_train_cmd = [
        "ns-train",
        "splatfacto",
        "--output-dir",
        str(output_dir),
        "--viewer.make-share-url",
        "False",
        "--viewer.quit-on-train-completion",
        "True",
        "--data",
        str(dataset_root),
        "nerfstudio-data",
        "--eval-mode",
        "fraction",
        "--train-split-fraction",
        "1.0",
        "--downscale-factor",
        str(downscale_factor),
    ]

    subprocess.run(ns_train_cmd, check=True)

    config_paths = sorted(
        output_dir.rglob("config.yml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert config_paths, f"ns-train did not create any configs under {output_dir}"
    return config_paths[0].parent


def _assert_checkpoint_exists(model_dir: Path) -> Path:
    """Refuse a run that stopped short of the 30K-iteration checkpoint the volumetric render loads.

    Args:
        model_dir: Run directory ns-train produced.

    Returns:
        Path of the step-000029999.ckpt checkpoint under model_dir / "nerfstudio_models".
    """
    checkpoint_path = model_dir / "nerfstudio_models" / f"step-000029999.ckpt"
    assert (
        checkpoint_path.is_file()
    ), f"Training did not reach 30K iterations; missing checkpoint {checkpoint_path}"
    return checkpoint_path
