import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.rotation.quaternion import (
    quat_to_rotmat,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
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
    """Rewrite one COLMAP reconstruction as a NerfStudio capture: a transforms record saved beside the sparse point cloud ply.

    Args:
        filename: File name of the transforms record, joined under `output_dir`.
        colmap_cameras: COLMAP camera records keyed by camera id, holding the one camera every image shares.
        colmap_images: COLMAP image records keyed by image id, each posed world-to-camera in the OpenCV convention by its `qvec` (w, x, y, z) and `tvec`.
        colmap_points: COLMAP 3D point records keyed by point id.
        output_dir: Directory the transforms record and the ply are written into, created when missing.
        ply_filename: File name of the sparse point cloud ply, joined under `output_dir` and recorded in the transforms record relative to it.
        pixel_error_filter: Reprojection error in pixels a point must stay under to enter the ply.
        point_track_filter: Number of images a point must be seen by to enter the ply.

    Returns:
        The path of the saved transforms record and the path of the written ply, in that order.
    """

    def _validate_inputs() -> None:
        assert isinstance(filename, str), f"{type(filename)=}"
        assert isinstance(colmap_cameras, dict), f"{type(colmap_cameras)=}"
        assert isinstance(colmap_images, dict), f"{type(colmap_images)=}"
        assert isinstance(colmap_points, dict), f"{type(colmap_points)=}"
        assert isinstance(output_dir, str), f"{type(output_dir)=}"
        assert isinstance(ply_filename, str), f"{type(ply_filename)=}"
        assert isinstance(pixel_error_filter, float), f"{type(pixel_error_filter)=}"
        assert isinstance(point_track_filter, int), f"{type(point_track_filter)=}"

    _validate_inputs()

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
    camera_intrinsics = cameras[0].intrinsics
    intrinsics = torch.tensor(
        [
            [camera_intrinsics.fx, 0.0, camera_intrinsics.cx],
            [0.0, camera_intrinsics.fy, camera_intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=cameras[0].device,
    )
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
    """Write the COLMAP sparse points surviving the reprojection-error / track-length filters as one ascii ply.

    Args:
        filename: File name of the ply, joined under `output_dir`.
        colmap_points: COLMAP 3D point records keyed by point id, each carrying `xyz`, `rgb`, `error` and the `image_ids` that see it.
        output_dir: Directory the ply is written into, created when missing.
        pixel_error_filter: Reprojection error in pixels a point must stay under to be written.
        point_track_filter: Number of images a point must be seen by to be written.

    Returns:
        The path of the written ply, whose vertices carry float x, y, z and uint8 red, green, blue.
    """

    def _validate_inputs() -> None:
        assert isinstance(filename, str), f"{type(filename)=}"
        assert isinstance(colmap_points, dict), f"{type(colmap_points)=}"
        assert isinstance(output_dir, str), f"{type(output_dir)=}"
        assert isinstance(pixel_error_filter, float), f"{type(pixel_error_filter)=}"
        assert isinstance(point_track_filter, int), f"{type(point_track_filter)=}"

    _validate_inputs()

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
            """Render one point as the ply vertex line that stands for it.

            Args:
                idx: Row index into the enclosing call's `points` / `colors` arrays.

            Returns:
                The point's x, y, z at eight decimals before its r, g, b as integers, space-separated and newline-closed.
            """
            coord = points[idx]
            color = colors[idx]
            x, y, z = coord
            r, g, b = color
            vertex_line = f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n"
            return vertex_line

        max_workers = min(32, len(valid_indices)) if len(valid_indices) > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            lines: List[str] = list(executor.map(_format_point, valid_indices))
        f.writelines(lines)

    return out_path


def _extract_intrinsics_from_colmap(
    colmap_cameras: Dict[int, ColmapCamera],
) -> Dict[str, Any]:
    """Read the one shared intrinsic set a COLMAP model records, restated in the undistorted OPENCV form NerfStudio writes.

    Args:
        colmap_cameras: COLMAP camera records keyed by camera id, holding exactly one camera whose model is SIMPLE_PINHOLE, PINHOLE, or distortion-free OPENCV.

    Returns:
        The intrinsic set keyed `w`, `h`, `fl_x`, `fl_y`, `cx`, `cy`, `k1`, `k2`, `p1`, `p2` (the four distortion entries all zero) and `camera_model` (`"OPENCV"`).
    """

    def _validate_inputs() -> None:
        assert isinstance(colmap_cameras, dict), f"{type(colmap_cameras)=}"
        assert colmap_cameras, "No cameras found in COLMAP model"
        assert (
            len(colmap_cameras) == 1
        ), f"Expected exactly one camera, got {len(colmap_cameras)}"
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

    _validate_inputs()

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
        assert False, (
            "Expected COLMAP camera model SIMPLE_PINHOLE, PINHOLE, or OPENCV. "
            f"{camera.model=}"
        )
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
    """Pose one Cameras out of the COLMAP images, every frame carrying the model's single shared intrinsic set.

    Args:
        colmap_images: COLMAP image records keyed by image id, each posed world-to-camera in the OpenCV convention by its `qvec` (w, x, y, z) and `tvec`.
        intrinsic_params: The shared intrinsic set `_extract_intrinsics_from_colmap` read, carrying `fl_x`, `fl_y`, `cx`, `cy`, `h`, `w`.

    Returns:
        A Cameras batch of one camera per image in ascending image-id order, named by image-file stem and identified by image id, with float32 standard-convention pinhole intrinsics and camera-to-world extrinsics in the OpenGL convention.
    """

    def _validate_inputs() -> None:
        assert isinstance(colmap_images, dict), f"{type(colmap_images)=}"
        assert colmap_images, "No images available in COLMAP model"
        assert isinstance(intrinsic_params, dict), f"{type(intrinsic_params)=}"

    _validate_inputs()

    intrinsics_params: Dict[str, Union[int, float]] = {
        "fx": intrinsic_params["fl_x"],
        "fy": intrinsic_params["fl_y"],
        "cx": intrinsic_params["cx"],
        "cy": intrinsic_params["cy"],
        "h": int(intrinsic_params["h"]),
        "w": int(intrinsic_params["w"]),
    }
    sorted_images = sorted(colmap_images.items())
    camera_ids: List[int] = [image_id for image_id, _ in sorted_images]
    camera_names: List[str] = [Path(image.name).stem for _, image in sorted_images]

    # The whole batch's pose stack is built in one op.
    quaternions = np.stack([image.qvec for _, image in sorted_images], axis=0)
    translations = np.stack([image.tvec for _, image in sorted_images], axis=0)
    assert np.issubdtype(quaternions.dtype, np.floating), (
        "Expected the COLMAP quaternions to be floating point before the float64 "
        f"cast. {quaternions.dtype=}"
    )
    assert np.issubdtype(translations.dtype, np.floating), (
        "Expected the COLMAP translations to be floating point before the float64 "
        f"cast. {translations.dtype=}"
    )
    # COLMAP states the pose world-to-camera, so cam2world is its rigid inverse: the transposed rotation, and that rotation applied to the negated translation.
    rotation = quat_to_rotmat(
        quaternions=torch.from_numpy(quaternions).to(torch.float64)
    ).transpose(-2, -1)
    translation = torch.from_numpy(translations).to(torch.float64)
    camera_to_world = torch.eye(4, dtype=torch.float64).repeat(len(sorted_images), 1, 1)
    camera_to_world[:, :3, :3] = rotation
    camera_to_world[:, :3, 3] = -(rotation @ translation.unsqueeze(-1)).squeeze(-1)
    extrinsics_opencv = camera_to_world.to(torch.float32)

    # One COLMAP camera governs every image, so its params broadcast to the batch.
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={
            key: torch.full(
                (len(sorted_images),),
                value,
                dtype=torch.float32,
                device=extrinsics_opencv.device,
            )
            for key, value in intrinsics_params.items()
        },
        intr_convention="standard",
        device=extrinsics_opencv.device,
    )
    extrinsics = CameraExtrinsics(
        extrinsics=extrinsics_opencv,
        extr_convention="opencv",
        device=extrinsics_opencv.device,
    )
    cameras = Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        names=camera_names,
        ids=camera_ids,
        device=extrinsics_opencv.device,
    )
    return cameras.to(extr_convention="opengl")


def _determine_modalities(cameras: Cameras, output_dir: Path) -> List[str]:
    """Name which of the image, depth, normal, mask modalities output_dir can serve for every camera.

    Args:
        cameras: The capture's cameras, every one named by its image-file stem.
        output_dir: The capture directory whose `depths` (.npy), `normals` (.png) and `masks` (.png) subdirectories are checked.

    Returns:
        The modality names: `image` first, then each of `depth`, `normal`, `mask` whose subdirectory holds a file for every camera name.
    """

    def _validate_inputs() -> None:
        assert isinstance(cameras, Cameras), f"{type(cameras)=}"
        assert len(cameras) > 0, "cameras must be non-empty"
        assert all(
            name is not None for name in cameras.names
        ), f"{list(cameras.names)=}"
        assert isinstance(output_dir, Path), f"{type(output_dir)=}"

    _validate_inputs()

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
