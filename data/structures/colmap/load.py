import collections
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

CAMERA_MODELS: Dict[int, Any] = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    5: CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    6: CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
}

CAMERA_MODEL_NAME_TO_ID: Dict[str, int] = {
    model.model_name: model_id for model_id, model in CAMERA_MODELS.items()
}

ColmapCamera = collections.namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"]
)

ColmapImage = collections.namedtuple(
    "ColmapImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

ColmapPoint3D = collections.namedtuple(
    "ColmapPoint3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def _load_next_bytes(
    fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"
) -> Tuple[Any, ...]:
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def load_colmap_data(
    model_dir: str | Path,
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

    path = Path(model_dir)
    assert path.is_dir(), f"COLMAP model dir not found: {path}"

    cameras_bin = path / "cameras.bin"
    images_bin = path / "images.bin"
    points_bin = path / "points3D.bin"
    cameras_txt = path / "cameras.txt"
    images_txt = path / "images.txt"
    points_txt = path / "points3D.txt"

    assert cameras_bin.exists() and images_bin.exists() and points_bin.exists(), (
        "COLMAP model must include binary files: " f"{path}"
    )
    assert cameras_txt.exists() and images_txt.exists() and points_txt.exists(), (
        "COLMAP model must include text files: " f"{path}"
    )

    return _load_colmap_data_bin(model_dir=path)


def _load_colmap_data_bin(
    model_dir: str | Path,
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

    path_obj = Path(model_dir)
    cameras_file = path_obj / "cameras.bin"
    images_file = path_obj / "images.bin"
    points3D_file = path_obj / "points3D.bin"

    assert cameras_file.exists(), f"cameras.bin not found: {cameras_file}"
    assert images_file.exists(), f"images.bin not found: {images_file}"
    assert points3D_file.exists(), f"points3D.bin not found: {points3D_file}"

    with ThreadPoolExecutor(max_workers=3) as executor:
        cameras_future = executor.submit(
            _load_colmap_cameras_bin, path_to_model_file=str(cameras_file)
        )
        images_future = executor.submit(
            _load_colmap_images_bin, path_to_model_file=str(images_file)
        )
        points_future = executor.submit(
            _load_colmap_points_bin, path_to_model_file=str(points3D_file)
        )
        cameras = cameras_future.result()
        images = images_future.result()
        points3D = points_future.result()
    return cameras, images, points3D


def _load_colmap_data_txt(
    model_dir: str | Path,
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

    path_obj = Path(model_dir)
    cameras_file = path_obj / "cameras.txt"
    images_file = path_obj / "images.txt"
    points3D_file = path_obj / "points3D.txt"

    assert cameras_file.exists(), f"cameras.txt not found: {cameras_file}"
    assert images_file.exists(), f"images.txt not found: {images_file}"
    assert points3D_file.exists(), f"points3D.txt not found: {points3D_file}"

    with ThreadPoolExecutor(max_workers=3) as executor:
        cameras_future = executor.submit(
            _load_colmap_cameras_txt, path_to_model_file=str(cameras_file)
        )
        images_future = executor.submit(
            _load_colmap_images_txt, path_to_model_file=str(images_file)
        )
        points_future = executor.submit(
            _load_colmap_points_txt, path_to_model_file=str(points3D_file)
        )
        cameras = cameras_future.result()
        images = images_future.result()
        points3D = points_future.result()
    return cameras, images, points3D


def _load_colmap_cameras_bin(path_to_model_file: str | Path) -> Dict[int, Any]:
    # Input validations
    assert isinstance(path_to_model_file, (str, Path)), f"{type(path_to_model_file)=}"

    cameras: Dict[int, Any] = {}
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

            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=CAMERA_MODELS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params),
            )

    return cameras


def _load_colmap_images_bin(path_to_model_file: str | Path) -> Dict[int, Any]:
    # Input validations
    assert isinstance(path_to_model_file, (str, Path)), f"{type(path_to_model_file)=}"

    images: Dict[int, Any] = {}
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
            while current_char != b"\x00":
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

            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )

    return images


def _load_colmap_points_bin(path_to_model_file: str | Path) -> Dict[int, Any]:
    # Input validations
    assert isinstance(path_to_model_file, (str, Path)), f"{type(path_to_model_file)=}"

    points3D: Dict[int, Any] = {}
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

            points3D[point3D_id] = ColmapPoint3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )

    return points3D


def _load_colmap_cameras_txt(path_to_model_file: str | Path) -> Dict[int, Any]:
    # Input validations
    assert isinstance(path_to_model_file, (str, Path)), f"{type(path_to_model_file)=}"

    path = Path(path_to_model_file)
    assert path.is_file(), f"cameras.txt not found: {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    content_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content_lines.append(stripped)
    assert content_lines, f"No camera entries found in {path}"
    cameras: Dict[int, Any] = {}
    max_workers = min(32, len(content_lines))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_parse_colmap_camera_line, content_lines)
        for camera_id, camera in results:
            assert (
                camera_id not in cameras
            ), f"Duplicate camera id {camera_id} in {path}"
            cameras[camera_id] = camera
    return cameras


def _load_colmap_images_txt(path_to_model_file: str | Path) -> Dict[int, Any]:
    # Input validations
    assert isinstance(path_to_model_file, (str, Path)), f"{type(path_to_model_file)=}"

    path = Path(path_to_model_file)
    assert path.is_file(), f"images.txt not found: {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    line_idx = 0
    blocks: List[Tuple[str, str]] = []
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line or line.startswith("#"):
            line_idx += 1
            continue
        header_line = line
        line_idx += 1
        assert line_idx < len(lines), f"Missing points2D line for image in {path}"
        points_line = lines[line_idx].strip()
        blocks.append((header_line, points_line))
        line_idx += 1
    assert blocks, f"No image entries found in {path}"
    images: Dict[int, Any] = {}
    max_workers = min(32, len(blocks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_parse_colmap_image_block, blocks)
        for image_id, image in results:
            assert image_id not in images, f"Duplicate image id {image_id} in {path}"
            images[image_id] = image
    return images


def _load_colmap_points_txt(path_to_model_file: str | Path) -> Dict[int, Any]:
    # Input validations
    assert isinstance(path_to_model_file, (str, Path)), f"{type(path_to_model_file)=}"

    path = Path(path_to_model_file)
    assert path.is_file(), f"points3D.txt not found: {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    content_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content_lines.append(stripped)
    assert content_lines, f"No points3D entries found in {path}"
    points3D: Dict[int, Any] = {}
    max_workers = min(32, len(content_lines))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_parse_colmap_point_line, content_lines)
        for point_id, point in results:
            assert point_id not in points3D, f"Duplicate point id {point_id} in {path}"
            points3D[point_id] = point
    return points3D


def _parse_colmap_camera_line(line: str) -> Tuple[int, ColmapCamera]:
    # Input validations
    assert isinstance(line, str), f"{type(line)=}"

    parts = line.split()
    assert len(parts) >= 5, f"Invalid cameras.txt line: {line}"
    camera_id = int(parts[0])
    model_name = parts[1]
    assert (
        model_name in CAMERA_MODEL_NAME_TO_ID
    ), f"Unknown camera model name {model_name}"
    width = int(parts[2])
    height = int(parts[3])
    params = np.array([float(val) for val in parts[4:]], dtype=np.float64)
    model_id = CAMERA_MODEL_NAME_TO_ID[model_name]
    expected_params = CAMERA_MODELS[model_id].num_params
    assert (
        params.size == expected_params
    ), f"Camera {camera_id} expected {expected_params} params, got {params.size}"
    return (
        camera_id,
        ColmapCamera(
            id=camera_id,
            model=model_name,
            width=width,
            height=height,
            params=params,
        ),
    )


def _parse_colmap_image_block(block: Tuple[str, str]) -> Tuple[int, ColmapImage]:
    # Input validations
    assert isinstance(block, tuple), f"{type(block)=}"
    assert len(block) == 2, f"{len(block)=}"
    assert isinstance(block[0], str), f"{type(block[0])=}"
    assert isinstance(block[1], str), f"{type(block[1])=}"

    header_line, points_line = block
    parts = header_line.split()
    assert len(parts) >= 10, f"Invalid images.txt line: {header_line}"
    image_id = int(parts[0])
    qvec = np.array(
        [
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        ],
        dtype=np.float64,
    )
    tvec = np.array(
        [
            float(parts[5]),
            float(parts[6]),
            float(parts[7]),
        ],
        dtype=np.float64,
    )
    camera_id = int(parts[8])
    name = " ".join(parts[9:])
    if points_line:
        point_parts = points_line.split()
        assert (
            len(point_parts) % 3 == 0
        ), f"Invalid points2D line for image {image_id}: {points_line}"
        num_points = len(point_parts) // 3
        xys = np.empty((num_points, 2), dtype=np.float64)
        point3d_ids = np.empty((num_points,), dtype=np.int64)
        for idx in range(num_points):
            base = idx * 3
            xys[idx, 0] = float(point_parts[base])
            xys[idx, 1] = float(point_parts[base + 1])
            point3d_ids[idx] = int(point_parts[base + 2])
    else:
        xys = np.empty((0, 2), dtype=np.float64)
        point3d_ids = np.empty((0,), dtype=np.int64)
    return (
        image_id,
        ColmapImage(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys,
            point3D_ids=point3d_ids,
        ),
    )


def _parse_colmap_point_line(line: str) -> Tuple[int, ColmapPoint3D]:
    # Input validations
    assert isinstance(line, str), f"{type(line)=}"

    parts = line.split()
    assert len(parts) >= 8, f"Invalid points3D.txt line: {line}"
    point_id = int(parts[0])
    xyz = np.array(
        [
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
        ],
        dtype=np.float64,
    )
    rgb = np.array(
        [
            int(parts[4]),
            int(parts[5]),
            int(parts[6]),
        ],
        dtype=np.uint8,
    )
    error = float(parts[7])
    track_parts = parts[8:]
    assert len(track_parts) % 2 == 0, f"Invalid track data for point {point_id}: {line}"
    track_len = len(track_parts) // 2
    image_ids = np.empty((track_len,), dtype=np.int32)
    point2d_idxs = np.empty((track_len,), dtype=np.int32)
    for idx in range(track_len):
        base = idx * 2
        image_ids[idx] = int(track_parts[base])
        point2d_idxs[idx] = int(track_parts[base + 1])
    return (
        point_id,
        ColmapPoint3D(
            id=point_id,
            xyz=xyz,
            rgb=rgb,
            error=error,
            image_ids=image_ids,
            point2D_idxs=point2d_idxs,
        ),
    )
