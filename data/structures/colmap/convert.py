import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np


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
