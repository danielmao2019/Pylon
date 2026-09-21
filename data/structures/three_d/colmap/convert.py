import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.rotation.quaternion import (
    quat_to_rotmat,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsics,
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

    def _colmap_to_pylon() -> Cameras:
        """Read the COLMAP cameras and images as the one posed Cameras Pylon holds them in, every image sharing the model's one intrinsics.

        Args:
            None; reads the `colmap_cameras` and `colmap_images` of the enclosing call.

        Returns:
            A Cameras batch of one camera per image in ascending image-id order, named by image-file stem and identified by image id, its one unbatched float32 standard-convention pinhole intrinsics broadcast over the images and its camera-to-world extrinsics in the OpenGL convention.
        """
        sorted_images = sorted(colmap_images.items())
        camera_ids: List[int] = []
        camera_names: List[str] = []
        images: List[ColmapImage] = []
        for image_id, image in sorted_images:
            camera_ids.append(image_id)
            camera_names.append(Path(image.name).stem)
            images.append(image)
        # Broadcast over every image, since one COLMAP camera governs them all.
        intrinsics = _extract_intrinsics_from_colmap(colmap_cameras=colmap_cameras)
        extrinsics = _extract_extrinsics_from_colmap(images=images)
        cameras = Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=camera_names,
            ids=camera_ids,
            device=extrinsics.device,
        )
        return cameras.to(extr_convention="opengl")

    cameras = _colmap_to_pylon()

    def _pylon_to_nerfstudio() -> None:
        """Write the posed Cameras as the NerfStudio transforms record at output_path, the ply beside it.

        Args:
            None; reads the `cameras`, `output_dir`, `output_path` and `ply_filename` of the enclosing call.

        Returns:
            None.
        """
        modalities = _determine_modalities(cameras=cameras, output_dir=Path(output_dir))
        # The batch's one intrinsics, no camera of it being the one read.
        camera_intrinsics = cameras.intrinsics
        # One, since the unbatched intrinsics is shared by every image.
        capture_params = {
            key: value.item() for key, value in camera_intrinsics.params.items()
        }
        # The undistorted OPENCV form NerfStudio writes.
        nerfstudio_intrinsic_params = {
            "fl_x": capture_params["fx"],
            "fl_y": capture_params["fy"],
            "cx": capture_params["cx"],
            "cy": capture_params["cy"],
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        }
        resolution = (int(capture_params["h"]), int(capture_params["w"]))
        camera_model = "OPENCV"
        intrinsics = torch.tensor(
            [
                [capture_params["fx"], 0.0, capture_params["cx"]],
                [0.0, capture_params["fy"], capture_params["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=cameras.device,
        )
        payload: Dict[str, Any] = {}
        nerfstudio_data = NerfStudio_Data(
            data=payload,
            device=cameras.device,
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
        return

    _pylon_to_nerfstudio()
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
            f.write(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 0\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uint8 red\n"
                "property uint8 green\n"
                "property uint8 blue\n"
                "end_header\n"
            )
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
        f.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_valid_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uint8 red\n"
            "property uint8 green\n"
            "property uint8 blue\n"
            "end_header\n"
        )

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

        max_workers = max(1, min(32, len(valid_indices)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            lines: List[str] = list(executor.map(_format_point, valid_indices))
        f.writelines(lines)

    return out_path


def _extract_intrinsics_from_colmap(
    colmap_cameras: Dict[int, ColmapCamera],
) -> CameraIntrinsics:
    """Read the one pinhole a COLMAP model records as the unbatched CameraIntrinsics every image shares.

    Args:
        colmap_cameras: COLMAP camera records keyed by camera id, holding exactly one camera whose model is SIMPLE_PINHOLE, PINHOLE, or distortion-free OPENCV, its width and height Python ints.

    Returns:
        The unbatched standard-convention pinhole CameraIntrinsics carrying the camera's `fx`, `fy`, `cx`, `cy` and its `h`, `w` resolution, its params 0-d tensors in the default float32 dtype.
    """

    def _validate_inputs() -> None:
        assert isinstance(colmap_cameras, dict), f"{type(colmap_cameras)=}"
        assert colmap_cameras, "No cameras found in COLMAP model"
        assert (
            len(colmap_cameras) == 1
        ), f"Expected exactly one camera, got {len(colmap_cameras)}"
        assert isinstance(
            next(iter(colmap_cameras.values())).width, int
        ), f"{type(next(iter(colmap_cameras.values())).width)=}"
        assert isinstance(
            next(iter(colmap_cameras.values())).height, int
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
        fl_x, fl_y = float(params[0]), float(params[0])
        cx, cy = float(params[1]), float(params[2])
    elif camera.model == "PINHOLE":
        assert len(params) == 4, f"Expected 4 params for PINHOLE, got {len(params)}"
        fl_x, fl_y = float(params[0]), float(params[1])
        cx, cy = float(params[2]), float(params[3])
    elif camera.model == "OPENCV":
        assert len(params) == 8, f"Expected 8 params for OPENCV, got {len(params)}"
        assert float(params[4]) == 0.0, f"k1 must be 0, got {params[4]}"
        assert float(params[5]) == 0.0, f"k2 must be 0, got {params[5]}"
        assert float(params[6]) == 0.0, f"p1 must be 0, got {params[6]}"
        assert float(params[7]) == 0.0, f"p2 must be 0, got {params[7]}"
        fl_x, fl_y = float(params[0]), float(params[1])
        cx, cy = float(params[2]), float(params[3])
    else:
        assert False, (
            "Expected COLMAP camera model SIMPLE_PINHOLE, PINHOLE, or OPENCV. "
            f"{camera.model=}"
        )
    width, height = camera.width, camera.height
    return build_camera_intrinsics(
        model="pinhole",
        params={"fx": fl_x, "fy": fl_y, "cx": cx, "cy": cy, "h": height, "w": width},
        intr_convention="standard",
    )


def _extract_extrinsics_from_colmap(images: List[ColmapImage]) -> CameraExtrinsics:
    """Pose the COLMAP images, in the order given, as one batched opencv CameraExtrinsics.

    Args:
        images: Non-empty list of COLMAP image records, each posed world-to-camera in the OpenCV convention by its floating `qvec` (w, x, y, z) and `tvec`.

    Returns:
        The batched CameraExtrinsics holding one float32 `[N, 4, 4]` camera-to-world matrix per image, in the order given, in the OpenCV convention.
    """

    def _validate_inputs() -> None:
        assert isinstance(images, list), f"{type(images)=}"
        assert len(images) > 0, "No images available in COLMAP model"

    _validate_inputs()

    qvecs, tvecs = [], []
    for image in images:
        qvecs.append(image.qvec)
        tvecs.append(image.tvec)
    # The whole batch's pose stack is built in one op.
    quaternions = np.stack(qvecs, axis=0)
    translations = np.stack(tvecs, axis=0)
    # Both stacks are cast to float64 next.
    assert np.issubdtype(quaternions.dtype, np.floating), (
        "Expected the COLMAP quaternions to be floating point before the float64 "
        f"cast. {quaternions.dtype=}"
    )
    assert np.issubdtype(translations.dtype, np.floating), (
        "Expected the COLMAP translations to be floating point before the float64 "
        f"cast. {translations.dtype=}"
    )
    # COLMAP states the pose world-to-camera, so cam2world is its rigid inverse.
    rotation = quat_to_rotmat(
        quaternions=torch.from_numpy(quaternions).to(torch.float64)
    ).transpose(-2, -1)
    translation = torch.from_numpy(translations).to(torch.float64)
    camera_to_world = torch.eye(4, dtype=torch.float64).repeat(len(images), 1, 1)
    camera_to_world[:, :3, :3] = rotation
    # The rigid inverse's translation: the transposed rotation applied to the negated translation.
    camera_to_world[:, :3, 3] = -(rotation @ translation.unsqueeze(-1)).squeeze(-1)
    extrinsics_opencv = camera_to_world.to(torch.float32)
    return CameraExtrinsics(
        extrinsics=extrinsics_opencv,
        extr_convention="opencv",
        device=extrinsics_opencv.device,
    )


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
