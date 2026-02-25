import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.rotation.quaternion import qvec2rotmat
from data.structures.three_d.colmap.load import ColmapCamera, ColmapImage
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data

DEFAULT_APPLIED_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _extract_intrinsics_from_colmap(
    colmap_cameras: Dict[int, ColmapCamera],
) -> Dict[str, Any]:
    # Input validations
    assert isinstance(colmap_cameras, dict), f"{type(colmap_cameras)=}"
    assert colmap_cameras, "No cameras found in COLMAP model"
    assert (
        len(colmap_cameras) == 1
    ), f"Expected exactly one camera, got {len(colmap_cameras)}"
    assert next(iter(colmap_cameras.values())).model in {
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "OPENCV",
    }, (
        "Expected COLMAP camera model SIMPLE_PINHOLE, PINHOLE, or OPENCV, "
        f"got {next(iter(colmap_cameras.values())).model}"
    )
    assert isinstance(
        next(iter(colmap_cameras.values())).width, (int, np.integer)
    ), f"{type(next(iter(colmap_cameras.values())).width)=}"
    assert isinstance(
        next(iter(colmap_cameras.values())).height, (int, np.integer)
    ), f"{type(next(iter(colmap_cameras.values())).height)=}"
    assert (
        next(iter(colmap_cameras.values())).width > 0
        and next(iter(colmap_cameras.values())).height > 0
    ), (
        "Camera dimensions must be positive, "
        f"got width={next(iter(colmap_cameras.values())).width} "
        f"height={next(iter(colmap_cameras.values())).height}"
    )

    camera = next(iter(colmap_cameras.values()))
    params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        assert (
            len(params) == 3
        ), f"Expected 3 params for SIMPLE_PINHOLE, got {len(params)}"
        fl_x = float(params[0])
        fl_y = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif camera.model == "PINHOLE":
        assert len(params) == 4, f"Expected 4 params for PINHOLE, got {len(params)}"
        fl_x = float(params[0])
        fl_y = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    elif camera.model == "OPENCV":
        assert len(params) == 8, f"Expected 8 params for OPENCV, got {len(params)}"
        assert float(params[4]) == 0.0, f"k1 must be 0, got {params[4]}"
        assert float(params[5]) == 0.0, f"k2 must be 0, got {params[5]}"
        assert float(params[6]) == 0.0, f"p1 must be 0, got {params[6]}"
        assert float(params[7]) == 0.0, f"p2 must be 0, got {params[7]}"
        fl_x = float(params[0])
        fl_y = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        assert False, f"Unsupported COLMAP camera model {camera.model}"
    width = camera.width
    height = camera.height
    intrinsic_params: Dict[str, Any] = {
        "w": int(width),
        "h": int(height),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
    }
    return intrinsic_params


def _extract_cameras_from_colmap(
    colmap_images: Dict[int, ColmapImage],
    intrinsic_params: Dict[str, Any],
) -> Cameras:
    # Input validations
    assert isinstance(colmap_images, dict), f"{type(colmap_images)=}"
    assert isinstance(intrinsic_params, dict), f"{type(intrinsic_params)=}"
    assert colmap_images, "No images available in COLMAP model"

    extrinsics_list: List[torch.Tensor] = []
    camera_names: List[str] = []
    camera_ids: List[int] = []
    intrinsics = torch.tensor(
        [
            [intrinsic_params["fl_x"], 0.0, intrinsic_params["cx"]],
            [0.0, intrinsic_params["fl_y"], intrinsic_params["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    intrinsics_list = [intrinsics for _ in colmap_images]
    for image_id, image in sorted(colmap_images.items()):
        rotation = qvec2rotmat(image.qvec)
        translation = image.tvec.reshape(3, 1)
        world_to_camera = np.concatenate([rotation, translation], axis=1)
        world_to_camera = np.concatenate(
            [world_to_camera, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        extrinsics_opencv = torch.from_numpy(camera_to_world).to(torch.float32)
        extrinsics_list.append(extrinsics_opencv)
        camera_names.append(Path(image.name).stem)
        camera_ids.append(image_id)
    assert extrinsics_list, "No cameras extracted from COLMAP images"
    cameras = Cameras(
        intrinsics=intrinsics_list,
        extrinsics=extrinsics_list,
        conventions=["opencv" for _ in extrinsics_list],
        names=camera_names,
        ids=camera_ids,
        device=extrinsics_list[0].device,
    )
    return cameras.to(convention="opengl")


def _determine_modalities(cameras: Cameras, output_dir: Path) -> List[str]:
    # Input validations
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert len(cameras) > 0, "cameras must be non-empty"
    assert isinstance(output_dir, Path), f"{type(output_dir)=}"
    assert all(name is not None for name in cameras.names), f"{list(cameras.names)=}"

    # Input normalizations
    camera_names = list(cameras.names)

    modalities = ["image"]

    depths_dir = output_dir / "depths"
    if depths_dir.is_dir():
        depth_names = {path.stem for path in depths_dir.glob("*.npy")}
        if set(camera_names).issubset(depth_names):
            modalities.append("depth")

    normals_dir = output_dir / "normals"
    if normals_dir.is_dir():
        normal_names = {path.stem for path in normals_dir.glob("*.png")}
        if set(camera_names).issubset(normal_names):
            modalities.append("normal")

    masks_dir = output_dir / "masks"
    if masks_dir.is_dir():
        mask_names = {path.stem for path in masks_dir.glob("*.png")}
        if set(camera_names).issubset(mask_names):
            modalities.append("mask")

    return modalities


def convert_colmap_to_nerfstudio(
    filename: str,
    colmap_cameras: Dict[int, ColmapCamera],
    colmap_images: Dict[int, ColmapImage],
    colmap_points: Dict[int, Any],
    output_dir: str,
    ply_filename: str = "sparse_pc.ply",
    pixel_error_filter: float = 1.0,
    point_track_filter: int = 5,
) -> Tuple[str, str]:
    # Input validations
    assert isinstance(filename, str), f"{type(filename)=}"
    assert isinstance(colmap_cameras, dict), f"{type(colmap_cameras)=}"
    assert isinstance(colmap_images, dict), f"{type(colmap_images)=}"
    assert isinstance(colmap_points, dict), f"{type(colmap_points)=}"
    assert isinstance(output_dir, str), f"{type(output_dir)=}"
    assert isinstance(ply_filename, str), f"{type(ply_filename)=}"
    assert isinstance(pixel_error_filter, float), f"{type(pixel_error_filter)=}"
    assert isinstance(point_track_filter, int), f"{type(point_track_filter)=}"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    ply_path = create_ply_from_colmap(
        filename=ply_filename,
        colmap_points=colmap_points,
        output_dir=output_dir,
        pixel_error_filter=pixel_error_filter,
        point_track_filter=point_track_filter,
    )

    intrinsic_params = _extract_intrinsics_from_colmap(colmap_cameras=colmap_cameras)
    cameras = _extract_cameras_from_colmap(
        colmap_images=colmap_images,
        intrinsic_params=intrinsic_params,
    )
    modalities = _determine_modalities(cameras=cameras, output_dir=Path(output_dir))

    nerfstudio_intrinsic_params = {
        "fl_x": intrinsic_params["fl_x"],
        "fl_y": intrinsic_params["fl_y"],
        "cx": intrinsic_params["cx"],
        "cy": intrinsic_params["cy"],
        "k1": intrinsic_params["k1"],
        "k2": intrinsic_params["k2"],
        "p1": intrinsic_params["p1"],
        "p2": intrinsic_params["p2"],
    }
    resolution = (intrinsic_params["h"], intrinsic_params["w"])
    camera_model = intrinsic_params["camera_model"]
    intrinsics = cameras[0].intrinsics
    payload: Dict[str, Any] = {}
    nerfstudio_data = NerfStudio_Data(
        data=payload,
        device=cameras[0].device,
        intrinsic_params=nerfstudio_intrinsic_params,
        resolution=resolution,
        camera_model=camera_model,
        intrinsics=intrinsics,
        applied_transform=DEFAULT_APPLIED_TRANSFORM,
        ply_file_path=ply_filename,
        cameras=cameras,
        modalities=modalities,
        train_filenames=None,
        val_filenames=None,
        test_filenames=None,
    )
    nerfstudio_data.save(output_path=output_path)
    return output_path, ply_path


def create_ply_from_colmap(
    filename: str,
    colmap_points: Dict[int, Any],
    output_dir: str,
    pixel_error_filter: float = 1.0,
    point_track_filter: int = 5,
) -> str:
    # Input validations
    assert isinstance(filename, str), f"{type(filename)=}"
    assert isinstance(colmap_points, dict), f"{type(colmap_points)=}"
    assert isinstance(output_dir, str), f"{type(output_dir)=}"
    assert isinstance(pixel_error_filter, float), f"{type(pixel_error_filter)=}"
    assert isinstance(point_track_filter, int), f"{type(point_track_filter)=}"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    if len(colmap_points) == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 0\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uint8 red\n")
            f.write("property uint8 green\n")
            f.write("property uint8 blue\n")
            f.write("end_header\n")
        return out_path

    point_ids = sorted(colmap_points)
    points = np.array([colmap_points[idx].xyz for idx in point_ids], dtype=np.float32)
    colors = np.array([colmap_points[idx].rgb for idx in point_ids], dtype=np.uint8)
    errors = np.array([colmap_points[idx].error for idx in point_ids], dtype=np.float32)
    track_lengths = np.array(
        [len(colmap_points[idx].image_ids) for idx in point_ids], dtype=np.uint8
    )

    valid_mask = np.logical_and(
        errors < pixel_error_filter, track_lengths >= point_track_filter
    )
    num_valid_points = int(valid_mask.sum())
    valid_indices = np.flatnonzero(valid_mask)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_valid_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")

        def _format_point(idx: int) -> str:
            # Input validations
            assert isinstance(idx, (int, np.integer)), f"{type(idx)=}"

            coord = points[idx]
            color = colors[idx]
            x, y, z = coord
            r, g, b = color
            return f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n"

        max_workers = min(32, len(valid_indices)) if len(valid_indices) > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            lines: List[str] = list(executor.map(_format_point, valid_indices))
        f.writelines(lines)

    return out_path
