"""Plotly rendering utilities for Gaussian-splatting PLY outputs."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData

_SH_DC_SCALE = 0.28209479177387814


def _rgb_to_css_color(
    rgb_values: np.ndarray,
    alpha_value: float,
) -> str:
    # Input validations
    assert isinstance(rgb_values, np.ndarray), f"{type(rgb_values)=}"
    assert isinstance(alpha_value, float), f"{type(alpha_value)=}"
    assert rgb_values.shape == (3,), f"{rgb_values.shape=}"
    assert 0.0 <= alpha_value <= 1.0, f"{alpha_value=}"

    rgb_uint8 = np.round(rgb_values.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_text = f"{alpha_value:.4f}"
    return (
        f"rgba({int(rgb_uint8[0])},{int(rgb_uint8[1])},{int(rgb_uint8[2])},"
        f"{alpha_text})"
    )


def _select_indices(
    num_items: int,
    max_items: int,
) -> np.ndarray:
    # Input validations
    assert isinstance(num_items, int), f"{type(num_items)=}"
    assert isinstance(max_items, int), f"{type(max_items)=}"
    assert num_items > 0, f"{num_items=}"
    assert max_items > 0, f"{max_items=}"

    if num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)

    sampled = np.linspace(0, num_items - 1, num=max_items)
    rounded = np.round(sampled).astype(np.int64)
    unique_indices = np.unique(rounded)
    assert unique_indices.shape[0] > 0, f"{unique_indices.shape=}"
    return unique_indices


def _normalize_rgb_values(rgb_values: np.ndarray) -> np.ndarray:
    # Input validations
    assert isinstance(rgb_values, np.ndarray), f"{type(rgb_values)=}"
    assert rgb_values.ndim == 2, f"{rgb_values.shape=}"
    assert rgb_values.shape[1] == 3, f"{rgb_values.shape=}"

    rgb_f32 = rgb_values.astype(np.float32)
    if float(rgb_f32.max()) > 1.0:
        rgb_f32 = rgb_f32 / 255.0
    return rgb_f32.clip(0.0, 1.0)


def _normalize_opacity_values(opacity_values: np.ndarray) -> np.ndarray:
    # Input validations
    assert isinstance(opacity_values, np.ndarray), f"{type(opacity_values)=}"
    assert opacity_values.ndim == 1, f"{opacity_values.shape=}"

    opacity_f32 = opacity_values.astype(np.float32)
    if float(opacity_f32.min()) < 0.0 or float(opacity_f32.max()) > 1.0:
        opacity_f32 = 1.0 / (1.0 + np.exp(-opacity_f32))
    return opacity_f32.clip(0.0, 1.0)


def _extract_scale_field_names(vertex_names: List[str]) -> List[str]:
    # Input validations
    assert isinstance(vertex_names, list), f"{type(vertex_names)=}"
    assert len(vertex_names) > 0, f"{len(vertex_names)=}"
    assert all(
        isinstance(field_name, str) for field_name in vertex_names
    ), f"{vertex_names=}"

    scale_names = [
        field_name for field_name in vertex_names if field_name.startswith("scale_")
    ]
    sorted_names = sorted(
        scale_names,
        key=lambda field_name: int(field_name.split("_")[1]),
    )
    return sorted_names


def _build_marker_sizes_from_scales(scale_values: np.ndarray) -> np.ndarray:
    # Input validations
    assert isinstance(scale_values, np.ndarray), f"{type(scale_values)=}"
    assert scale_values.ndim == 2, f"{scale_values.shape=}"
    assert scale_values.shape[1] >= 1, f"{scale_values.shape=}"

    scale_exp = np.exp(scale_values.astype(np.float32))
    scale_mean = scale_exp.mean(axis=1)
    assert scale_mean.ndim == 1, f"{scale_mean.shape=}"
    assert (
        scale_mean.shape[0] == scale_values.shape[0]
    ), f"{scale_mean.shape=} {scale_values.shape=}"

    scale_ref = float(np.percentile(scale_mean, 95.0))
    assert scale_ref > 0.0, f"{scale_ref=}"
    normalized = scale_mean / scale_ref
    marker_sizes = (normalized * 8.0).clip(1.5, 12.0)
    return marker_sizes.astype(np.float32)


def _parse_gaussian_payload(
    vertex_table: Any,
    selected_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    # Input validations
    assert isinstance(selected_indices, np.ndarray), f"{type(selected_indices)=}"
    assert selected_indices.ndim == 1, f"{selected_indices.shape=}"
    assert selected_indices.shape[0] > 0, f"{selected_indices.shape=}"

    selected_rows = vertex_table[selected_indices]
    positions = np.stack(
        [
            selected_rows["x"].astype(np.float32),
            selected_rows["y"].astype(np.float32),
            selected_rows["z"].astype(np.float32),
        ],
        axis=1,
    )
    assert positions.shape[0] > 0, f"{positions.shape=}"
    assert positions.shape[1] == 3, f"{positions.shape=}"

    if {"r", "g", "b"}.issubset(selected_rows.dtype.names):
        rgb_values = np.stack(
            [
                selected_rows["r"].astype(np.float32),
                selected_rows["g"].astype(np.float32),
                selected_rows["b"].astype(np.float32),
            ],
            axis=1,
        )
        colors = _normalize_rgb_values(rgb_values=rgb_values)
    else:
        assert {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(
            selected_rows.dtype.names
        ), f"{selected_rows.dtype.names=}"
        colors = np.stack(
            [
                selected_rows["f_dc_0"].astype(np.float32) * _SH_DC_SCALE + 0.5,
                selected_rows["f_dc_1"].astype(np.float32) * _SH_DC_SCALE + 0.5,
                selected_rows["f_dc_2"].astype(np.float32) * _SH_DC_SCALE + 0.5,
            ],
            axis=1,
        ).clip(0.0, 1.0)

    opacities = _normalize_opacity_values(
        opacity_values=selected_rows["opacity"].astype(np.float32),
    )

    vertex_names = list(selected_rows.dtype.names)
    scale_field_names = _extract_scale_field_names(vertex_names=vertex_names)
    assert len(scale_field_names) >= 1, f"{scale_field_names=}"
    scale_values = np.stack(
        [
            selected_rows[field_name].astype(np.float32)
            for field_name in scale_field_names
        ],
        axis=1,
    )
    marker_sizes = _build_marker_sizes_from_scales(scale_values=scale_values)
    return {
        "structure_type": np.asarray(["gaussian_splatting"]),
        "positions": positions,
        "colors": colors,
        "opacities": opacities,
        "marker_sizes": marker_sizes,
    }


@lru_cache(maxsize=128)
def _load_gaussian_payload_cached(
    ply_path_str: str,
    max_primitives: int,
) -> Dict[str, np.ndarray]:
    # Input validations
    assert isinstance(ply_path_str, str), f"{type(ply_path_str)=}"
    assert isinstance(max_primitives, int), f"{type(max_primitives)=}"
    assert ply_path_str != "", f"{ply_path_str=}"
    assert max_primitives > 0, f"{max_primitives=}"

    ply_path = Path(ply_path_str)
    assert ply_path.exists(), f"{ply_path=}"
    assert ply_path.is_file(), f"{ply_path=}"

    ply_data = PlyData.read(str(ply_path))
    assert "vertex" in ply_data, f"{ply_data=}"
    vertex_element = ply_data["vertex"]
    vertex_table = vertex_element.data
    vertex_names = list(vertex_table.dtype.names)
    assert {"x", "y", "z"}.issubset(vertex_names), f"{vertex_names=}"
    assert vertex_element.count > 0, f"{vertex_element.count=}"

    scale_field_names = _extract_scale_field_names(vertex_names=vertex_names)
    assert "opacity" in vertex_names, f"{vertex_names=}"
    assert len(scale_field_names) >= 1, f"{scale_field_names=}"
    assert any(
        field_name.startswith("rot_") for field_name in vertex_names
    ), f"{vertex_names=}"
    assert {"r", "g", "b"}.issubset(vertex_names) or {
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }.issubset(vertex_names), f"{vertex_names=}"

    selected_indices = _select_indices(
        num_items=vertex_element.count,
        max_items=max_primitives,
    )
    assert selected_indices.shape[0] > 0, f"{selected_indices.shape=}"

    payload = _parse_gaussian_payload(
        vertex_table=vertex_table,
        selected_indices=selected_indices,
    )
    payload["full_count"] = np.asarray([int(vertex_element.count)], dtype=np.int64)
    payload["display_count"] = np.asarray(
        [int(selected_indices.shape[0])],
        dtype=np.int64,
    )
    return payload


def _build_scene_camera(
    source_camera_center: np.ndarray,
    source_camera_up: np.ndarray,
    geometry_center: np.ndarray,
    camera_override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Input validations
    assert isinstance(
        source_camera_center,
        np.ndarray,
    ), f"{type(source_camera_center)=}"
    assert isinstance(source_camera_up, np.ndarray), f"{type(source_camera_up)=}"
    assert isinstance(geometry_center, np.ndarray), f"{type(geometry_center)=}"
    assert camera_override is None or isinstance(
        camera_override,
        dict,
    ), f"{type(camera_override)=}"
    assert source_camera_center.shape == (3,), f"{source_camera_center.shape=}"
    assert source_camera_up.shape == (3,), f"{source_camera_up.shape=}"
    assert geometry_center.shape == (3,), f"{geometry_center.shape=}"

    if camera_override is not None:
        return camera_override

    centered_eye = source_camera_center - geometry_center
    return {
        "eye": {
            "x": float(centered_eye[0]),
            "y": float(centered_eye[1]),
            "z": float(centered_eye[2]),
        },
        "up": {
            "x": float(source_camera_up[0]),
            "y": float(source_camera_up[1]),
            "z": float(source_camera_up[2]),
        },
    }


def load_gaussian_splatting_metadata(
    structure_ply_path: Path,
    max_primitives: int,
) -> Dict[str, Any]:
    # Input validations
    assert isinstance(structure_ply_path, Path), f"{type(structure_ply_path)=}"
    assert isinstance(max_primitives, int), f"{type(max_primitives)=}"
    assert structure_ply_path.exists(), f"{structure_ply_path=}"
    assert structure_ply_path.is_file(), f"{structure_ply_path=}"
    assert max_primitives > 0, f"{max_primitives=}"

    payload = _load_gaussian_payload_cached(
        ply_path_str=str(structure_ply_path.resolve()),
        max_primitives=max_primitives,
    )
    return {
        "structure_type": str(payload["structure_type"][0]),
        "full_count": int(payload["full_count"][0]),
        "display_count": int(payload["display_count"][0]),
    }


def build_gaussian_splatting_figure(
    structure_ply_path: Path,
    source_camera_center: np.ndarray,
    source_camera_up: np.ndarray,
    camera_override: Optional[Dict[str, Any]],
    uirevision_key: str,
    max_primitives: int,
) -> go.Figure:
    # Input validations
    assert isinstance(structure_ply_path, Path), f"{type(structure_ply_path)=}"
    assert isinstance(
        source_camera_center,
        np.ndarray,
    ), f"{type(source_camera_center)=}"
    assert isinstance(source_camera_up, np.ndarray), f"{type(source_camera_up)=}"
    assert camera_override is None or isinstance(
        camera_override,
        dict,
    ), f"{type(camera_override)=}"
    assert isinstance(uirevision_key, str), f"{type(uirevision_key)=}"
    assert isinstance(max_primitives, int), f"{type(max_primitives)=}"
    assert structure_ply_path.exists(), f"{structure_ply_path=}"
    assert structure_ply_path.is_file(), f"{structure_ply_path=}"
    assert source_camera_center.shape == (3,), f"{source_camera_center.shape=}"
    assert source_camera_up.shape == (3,), f"{source_camera_up.shape=}"
    assert uirevision_key != "", f"{uirevision_key=}"
    assert max_primitives > 0, f"{max_primitives=}"

    payload = _load_gaussian_payload_cached(
        ply_path_str=str(structure_ply_path.resolve()),
        max_primitives=max_primitives,
    )
    positions = payload["positions"]
    colors = payload["colors"]
    opacities = payload["opacities"]
    marker_sizes = payload["marker_sizes"]

    assert positions.ndim == 2, f"{positions.shape=}"
    assert positions.shape[1] == 3, f"{positions.shape=}"
    assert positions.shape[0] > 0, f"{positions.shape=}"
    assert colors.shape == positions.shape, f"{colors.shape=} {positions.shape=}"
    assert opacities.shape == (positions.shape[0],), f"{opacities.shape=}"
    assert marker_sizes.shape == (positions.shape[0],), f"{marker_sizes.shape=}"

    geometry_center = positions.mean(axis=0)
    assert geometry_center.shape == (3,), f"{geometry_center.shape=}"
    centered_positions = positions - geometry_center.reshape(1, 3)
    scene_camera = _build_scene_camera(
        source_camera_center=source_camera_center,
        source_camera_up=source_camera_up,
        geometry_center=geometry_center,
        camera_override=camera_override,
    )
    marker_colors = [
        _rgb_to_css_color(
            rgb_values=colors[idx],
            alpha_value=float(opacities[idx]),
        )
        for idx in range(positions.shape[0])
    ]

    point_trace = go.Scatter3d(
        x=centered_positions[:, 0],
        y=centered_positions[:, 1],
        z=centered_positions[:, 2],
        mode="markers",
        marker={
            "size": marker_sizes,
            "color": marker_colors,
        },
        hoverinfo="skip",
        name="gaussian_splatting",
    )
    figure = go.Figure(data=[point_trace])
    figure.update_layout(
        uirevision=uirevision_key,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#ffffff",
        scene={
            "bgcolor": "#f8fafc",
            "aspectmode": "data",
            "uirevision": uirevision_key,
            "camera": scene_camera,
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False},
        },
    )
    return figure
