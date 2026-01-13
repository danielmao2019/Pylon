"""
COLMAP utilities for loading and processing reconstruction data.

This module provides functionality to load COLMAP binary reconstruction files
and extract camera poses, intrinsics, and 3D points.

Based on COLMAP's official loading utilities with support for multiple camera models
and quaternion/rotation matrix conversions.
"""

import collections
import os
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from utils.io.colmap.camera_models import (
    CAMERA_MODELS,
    CAMERA_MODEL_NAME_TO_ID,
    CameraModel,
)
from utils.three_d.rotation.quaternion import qvec2rotmat, rotmat2qvec

# COLMAP data structures
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def _load_next_bytes(
    fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"
) -> tuple:
    """Load and unpack the next bytes from a binary file.

    Args:
        fid: File handle
        num_bytes: Number of bytes to read
        format_char_sequence: Struct format string
        endian_character: Endianness character

    Returns:
        Unpacked data
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def load_cameras_binary(path_to_model_file: str) -> Dict[int, Camera]:
    """Load cameras from COLMAP binary file.

    Args:
        path_to_model_file: Path to cameras.bin file

    Returns:
        Dictionary mapping camera ID to Camera object
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = _load_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = _load_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]

            num_params = CAMERA_MODELS[model_id].num_params
            params = _load_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )

            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODELS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params),
            )

    return cameras


def load_images_binary(path_to_model_file: str) -> Dict[int, Image]:
    """Load images from COLMAP binary file.

    Args:
        path_to_model_file: Path to images.bin file

    Returns:
        Dictionary mapping image ID to Image object
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = _load_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = _load_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            image_name = ""
            current_char = _load_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = _load_next_bytes(fid, 1, "c")[0]

            num_points2D = _load_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = _load_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )

            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )

    return images


def load_points3D_binary(path_to_model_file: str) -> Dict[int, Point3D]:
    """Load 3D points from COLMAP binary file.

    Args:
        path_to_model_file: Path to points3D.bin file

    Returns:
        Dictionary mapping point ID to Point3D object
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = _load_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = _load_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])

            track_length = _load_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = _load_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )

            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))

            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )

    return points3D


def load_model(
    path: str,
) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """Load COLMAP model from binary files.

    Args:
        path: Path to directory containing cameras.bin, images.bin, points3D.bin

    Returns:
        Tuple of (cameras, images, points3D) dictionaries
    """
    path_obj = Path(path)

    cameras_file = path_obj / "cameras.bin"
    images_file = path_obj / "images.bin"
    points3D_file = path_obj / "points3D.bin"

    assert cameras_file.exists(), f"cameras.bin not found: {cameras_file}"
    assert images_file.exists(), f"images.bin not found: {images_file}"
    assert points3D_file.exists(), f"points3D.bin not found: {points3D_file}"

    cameras = load_cameras_binary(str(cameras_file))
    images = load_images_binary(str(images_file))
    points3D = load_points3D_binary(str(points3D_file))

    return cameras, images, points3D


def get_camera_positions(images: Dict[int, Image]) -> np.ndarray:
    """Extract camera positions from COLMAP images.

    Args:
        images: Dictionary of COLMAP images

    Returns:
        Array of camera positions (N, 3)
    """
    positions = []
    for img in images.values():
        R = qvec2rotmat(img.qvec).T  # Transpose for world-to-camera to camera-to-world
        t = np.array(img.tvec)
        position = -R @ t  # Camera position in world coordinates
        positions.append(position)

    return np.array(positions)


def create_ply_from_colmap(
    filename: str,
    colmap_points: Dict[int, Point3D],
    output_dir: str,
    pixel_error_filter: float = 1.0,
    point_track_filter: int = 5,
) -> str:
    """Write a filtered COLMAP sparse point cloud to a PLY file.

    Args:
        filename: Output filename for the PLY file
        colmap_points: Dictionary of COLMAP Point3D entries
        output_dir: Directory to write the PLY file into
        pixel_error_filter: Maximum reprojection error to keep a point
        point_track_filter: Minimum number of observations to keep a point

    Returns:
        Path to the written PLY file
    """

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

    points = np.array([p.xyz for p in colmap_points.values()], dtype=np.float32)
    colors = np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8)
    errors = np.array([p.error for p in colmap_points.values()], dtype=np.float32)
    track_lengths = np.array(
        [len(p.image_ids) for p in colmap_points.values()], dtype=np.uint8
    )

    valid_mask = np.logical_and(
        errors < pixel_error_filter, track_lengths >= point_track_filter
    )
    num_valid_points = int(valid_mask.sum())

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

        for coord, color, is_valid in zip(points, colors, valid_mask):
            if not is_valid:
                continue
            x, y, z = coord
            r, g, b = color
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")

    return out_path
