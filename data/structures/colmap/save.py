"""Generic COLMAP save utilities for cameras, images, and sparse points."""

import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from data.structures.colmap.load import CAMERA_MODEL_NAME_TO_ID, CAMERA_MODELS


def save_colmap_data(
    output_dir: str | Path,
    cameras: Dict[int, Any],
    images: Dict[int, Any],
    points3D: Dict[int, Any],
) -> None:
    # Input validations
    assert isinstance(output_dir, (str, Path)), f"{type(output_dir)=}"
    assert isinstance(cameras, dict), f"{type(cameras)=}"
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(points3D, dict), f"{type(points3D)=}"

    _save_colmap_data_bin(
        output_dir=output_dir,
        cameras=cameras,
        images=images,
        points3D=points3D,
    )
    _save_colmap_data_txt(
        output_dir=output_dir,
        cameras=cameras,
        images=images,
        points3D=points3D,
    )


def _save_colmap_data_bin(
    output_dir: str | Path,
    cameras: Dict[int, Any],
    images: Dict[int, Any],
    points3D: Dict[int, Any],
) -> None:
    # Input validations
    assert isinstance(output_dir, (str, Path)), f"{type(output_dir)=}"
    assert isinstance(cameras, dict), f"{type(cameras)=}"
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(points3D, dict), f"{type(points3D)=}"

    output_root = Path(output_dir)
    with ThreadPoolExecutor(max_workers=3) as executor:
        cameras_future = executor.submit(
            _save_colmap_cameras_bin,
            output_path=output_root / "cameras.bin",
            cameras=cameras,
        )
        images_future = executor.submit(
            _save_colmap_images_bin,
            output_path=output_root / "images.bin",
            images=images,
        )
        points_future = executor.submit(
            _save_colmap_points_bin,
            output_path=output_root / "points3D.bin",
            points3D=points3D,
        )
        cameras_future.result()
        images_future.result()
        points_future.result()


def _save_colmap_data_txt(
    output_dir: str | Path,
    cameras: Dict[int, Any],
    images: Dict[int, Any],
    points3D: Dict[int, Any],
) -> None:
    # Input validations
    assert isinstance(output_dir, (str, Path)), f"{type(output_dir)=}"
    assert isinstance(cameras, dict), f"{type(cameras)=}"
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(points3D, dict), f"{type(points3D)=}"

    output_root = Path(output_dir)
    with ThreadPoolExecutor(max_workers=3) as executor:
        cameras_future = executor.submit(
            _save_colmap_cameras_txt,
            output_path=output_root / "cameras.txt",
            cameras=cameras,
        )
        images_future = executor.submit(
            _save_colmap_images_txt,
            output_path=output_root / "images.txt",
            images=images,
        )
        points_future = executor.submit(
            _save_colmap_points_txt,
            output_path=output_root / "points3D.txt",
            points3D=points3D,
        )
        cameras_future.result()
        images_future.result()
        points_future.result()


def _save_colmap_cameras_bin(output_path: str | Path, cameras: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"
    assert isinstance(cameras, dict), f"{type(cameras)=}"

    assert cameras, "No cameras provided for cameras.bin"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            assert (
                camera.id == camera_id
            ), f"Camera id key {camera_id} does not match stored id {camera.id}"
            assert (
                camera.model in CAMERA_MODEL_NAME_TO_ID
            ), f"Unknown camera model {camera.model}"
            model_id = CAMERA_MODEL_NAME_TO_ID[camera.model]
            expected_params = CAMERA_MODELS[model_id].num_params
            params = np.asarray(camera.params, dtype=np.float64)
            assert (
                params.size == expected_params
            ), f"Camera {camera_id} expected {expected_params} params, got {params.size}"
            width = int(camera.width)
            height = int(camera.height)
            assert (
                width > 0 and height > 0
            ), f"Camera dimensions must be positive for {camera_id}"
            handle.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
            handle.write(struct.pack("<" + "d" * expected_params, *params))


def _save_colmap_images_bin(output_path: str | Path, images: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"
    assert isinstance(images, dict), f"{type(images)=}"

    assert images, "No images provided for images.bin"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image_id in sorted(images):
            image = images[image_id]
            assert (
                image.id == image_id
            ), f"Image id key {image_id} does not match stored id {image.id}"
            qvec = np.asarray(image.qvec, dtype=np.float64)
            tvec = np.asarray(image.tvec, dtype=np.float64).reshape(-1)
            assert qvec.shape == (4,), f"qvec for image {image_id} must have shape (4,)"
            assert tvec.shape == (3,), f"tvec for image {image_id} must have shape (3,)"
            name_bytes = image.name.encode("utf-8")
            assert 0 not in name_bytes, f"image name for {image_id} contains null byte"
            xys = np.asarray(image.xys, dtype=np.float64)
            point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
            assert (
                xys.shape[0] == point_ids.shape[0]
            ), f"image {image_id} has mismatched xys and point ids"
            handle.write(
                struct.pack(
                    "<idddddddi",
                    image_id,
                    qvec[0],
                    qvec[1],
                    qvec[2],
                    qvec[3],
                    tvec[0],
                    tvec[1],
                    tvec[2],
                    int(image.camera_id),
                )
            )
            handle.write(name_bytes + b"\x00")
            handle.write(struct.pack("<Q", xys.shape[0]))
            for xy, point_id in zip(xys, point_ids, strict=True):
                assert xy.shape == (
                    2,
                ), f"xy entry for image {image_id} must have shape (2,)"
                handle.write(struct.pack("<ddq", xy[0], xy[1], int(point_id)))


def _save_colmap_points_bin(output_path: str | Path, points3D: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"
    assert isinstance(points3D, dict), f"{type(points3D)=}"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points3D)))
        for point_id in sorted(points3D):
            point = points3D[point_id]
            assert (
                point.id == point_id
            ), f"Point id key {point_id} does not match stored id {point.id}"
            coords = np.asarray(point.xyz, dtype=np.float64)
            assert coords.shape == (
                3,
            ), f"xyz for point {point_id} must have shape (3,)"
            color = np.asarray(point.rgb, dtype=np.uint8).reshape(-1)
            assert color.shape == (3,), f"rgb for point {point_id} must have shape (3,)"
            error_value = float(point.error)
            image_ids = np.asarray(point.image_ids, dtype=np.int32).reshape(-1)
            point2d_ids = np.asarray(point.point2D_idxs, dtype=np.int32).reshape(-1)
            assert (
                image_ids.shape == point2d_ids.shape
            ), f"Track lengths mismatch for point {point_id}"
            handle.write(
                struct.pack(
                    "<QdddBBBd",
                    int(point_id),
                    coords[0],
                    coords[1],
                    coords[2],
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    error_value,
                )
            )
            track_length = image_ids.shape[0]
            handle.write(struct.pack("<Q", track_length))
            for image_idx, point2d_idx in zip(image_ids, point2d_ids, strict=True):
                handle.write(struct.pack("<ii", int(image_idx), int(point2d_idx)))


def _save_colmap_cameras_txt(output_path: str | Path, cameras: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"
    assert isinstance(cameras, dict), f"{type(cameras)=}"

    assert cameras, "No cameras provided for cameras.txt"
    path = Path(output_path)
    lines: List[str] = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for camera_id in sorted(cameras):
        camera = cameras[camera_id]
        assert (
            camera.id == camera_id
        ), f"Camera id key {camera_id} does not match stored id {camera.id}"
        assert (
            camera.model in CAMERA_MODEL_NAME_TO_ID
        ), f"Unknown camera model {camera.model}"
        model_id = CAMERA_MODEL_NAME_TO_ID[camera.model]
        expected_params = CAMERA_MODELS[model_id].num_params
        params = np.asarray(camera.params, dtype=np.float64)
        assert (
            params.size == expected_params
        ), f"Camera {camera_id} expected {expected_params} params, got {params.size}"
        width = int(camera.width)
        height = int(camera.height)
        assert (
            width > 0 and height > 0
        ), f"Camera dimensions must be positive for {camera_id}"
        params_str = " ".join(f"{float(param):.17g}" for param in params)
        lines.append(f"{camera_id} {camera.model} {width} {height} {params_str}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_colmap_images_txt(output_path: str | Path, images: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"
    assert isinstance(images, dict), f"{type(images)=}"

    assert images, "No images provided for images.txt"
    path = Path(output_path)
    lines: List[str] = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id in sorted(images):
        image = images[image_id]
        assert (
            image.id == image_id
        ), f"Image id key {image_id} does not match stored id {image.id}"
        qvec = np.asarray(image.qvec, dtype=np.float64)
        assert qvec.shape == (4,), f"qvec for image {image_id} must have shape (4,)"
        tvec = np.asarray(image.tvec, dtype=np.float64).reshape(-1)
        assert tvec.shape == (3,), f"tvec for image {image_id} must have shape (3,)"
        xys = np.asarray(image.xys, dtype=np.float64)
        point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
        assert (
            xys.shape[0] == point_ids.shape[0]
        ), f"image {image_id} has mismatched xys and point ids"
        header = (
            f"{image_id} "
            f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
            f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} "
            f"{image.camera_id} {image.name}"
        )
        lines.append(header)
        if xys.size == 0:
            lines.append("")
            continue
        track_parts: List[str] = []
        for xy, point_id in zip(xys, point_ids, strict=True):
            assert xy.shape == (
                2,
            ), f"xy entry for image {image_id} must have shape (2,)"
            track_parts.append(f"{xy[0]:.17g} {xy[1]:.17g} {int(point_id)}")
        lines.append(" ".join(track_parts))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_colmap_points_txt(output_path: str | Path, points3D: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"
    assert isinstance(points3D, dict), f"{type(points3D)=}"

    path = Path(output_path)
    lines: List[str] = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for point_id in sorted(points3D):
        point = points3D[point_id]
        assert (
            point.id == point_id
        ), f"Point id key {point_id} does not match stored id {point.id}"
        coords = np.asarray(point.xyz, dtype=np.float64)
        assert coords.shape == (3,), f"xyz for point {point_id} must have shape (3,)"
        color = np.asarray(point.rgb, dtype=np.uint8).reshape(-1)
        assert color.shape == (3,), f"rgb for point {point_id} must have shape (3,)"
        image_ids = np.asarray(point.image_ids, dtype=np.int32).reshape(-1)
        point2d_ids = np.asarray(point.point2D_idxs, dtype=np.int32).reshape(-1)
        assert (
            image_ids.shape == point2d_ids.shape
        ), f"Track lengths mismatch for point {point_id}"
        track_parts = []
        for image_idx, point2d_idx in zip(image_ids, point2d_ids, strict=True):
            track_parts.append(f"{int(image_idx)} {int(point2d_idx)}")
        track_str = " ".join(track_parts)
        lines.append(
            f"{point_id} "
            f"{coords[0]:.17g} {coords[1]:.17g} {coords[2]:.17g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} "
            f"{float(point.error):.17g}" + (f" {track_str}" if track_str else "")
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
