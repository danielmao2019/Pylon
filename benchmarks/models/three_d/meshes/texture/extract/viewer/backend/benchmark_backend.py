"""Backend helpers for the texture-extraction benchmark viewer."""

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.meshes.texture.extract.extract import (
    _build_uv_rasterization_data,
    _compute_f_normals_weights,
    _project_f_colors,
    _rasterize_face_weights_to_uv,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility import (
    compute_f_visibility_mask,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import (
    compute_f_visibility_mask_v2,
)
from models.three_d.meshes.texture.extract.weights_cfg import normalize_weights_cfg

DEFAULT_PROCESSED_GSO_ROOT = Path("/pub0/data/clod_processed/google_scanned_objects")
DEFAULT_RAW_GSO_ROOT = Path("/pub0/data/clod/google_scanned_objects")
DEFAULT_SCENE_NAME = "3D_Dollhouse_Lamp"
DEFAULT_WEIGHTS_CFG: Dict[str, Any] = {
    "weights": "normals",
    "normals_weight_power": 1.0,
    "normals_weight_threshold": 0.0,
}
REPETITIONS = 3
WARMUP_RUNS = 1


def get_results_root() -> Path:
    """Return the results root for this benchmark package.

    Args:
        None.

    Returns:
        Results-root path.
    """

    return Path(__file__).resolve().parents[2] / "results"


def list_scene_names(
    processed_gso_root: Path = DEFAULT_PROCESSED_GSO_ROOT,
) -> List[str]:
    """List all processed `gso-clod` scene names for this benchmark.

    Args:
        processed_gso_root: Root directory that contains processed `gso-clod` scenes.

    Returns:
        Sorted scene-name list.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(processed_gso_root, Path), (
            "Expected `processed_gso_root` to be a `Path`. "
            f"{type(processed_gso_root)=}."
        )
        assert processed_gso_root.exists(), (
            "Expected `processed_gso_root` to exist. " f"{processed_gso_root=}"
        )
        assert processed_gso_root.is_dir(), (
            "Expected `processed_gso_root` to be a directory. " f"{processed_gso_root=}"
        )

    _validate_inputs()

    scene_names = sorted(
        child.name
        for child in processed_gso_root.iterdir()
        if child.is_dir() and (child / "transforms.json").is_file()
    )
    assert len(scene_names) == 30, (
        "Expected the `gso-clod` benchmark to cover exactly 30 scenes. "
        f"{len(scene_names)=}."
    )
    return scene_names


def prepare_benchmark_results(
    results_root: Path,
    force: bool = False,
    processed_gso_root: Path = DEFAULT_PROCESSED_GSO_ROOT,
    raw_gso_root: Path = DEFAULT_RAW_GSO_ROOT,
) -> Dict[str, Any]:
    """Prepare the full cached benchmark results for all scenes.

    Args:
        results_root: Benchmark results root.
        force: Whether to rebuild all per-scene results even if cached.
        processed_gso_root: Root directory that contains processed `gso-clod` scenes.
        raw_gso_root: Root directory that contains raw GSO mesh assets.

    Returns:
        Results-index dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )
        assert isinstance(force, bool), (
            "Expected `force` to be a bool. " f"{type(force)=}."
        )
        assert isinstance(processed_gso_root, Path), (
            "Expected `processed_gso_root` to be a `Path`. "
            f"{type(processed_gso_root)=}."
        )
        assert isinstance(raw_gso_root, Path), (
            "Expected `raw_gso_root` to be a `Path`. " f"{type(raw_gso_root)=}."
        )

    _validate_inputs()

    results_root.mkdir(parents=True, exist_ok=True)
    scene_names = list_scene_names(processed_gso_root=processed_gso_root)
    scene_summaries: Dict[str, Dict[str, Any]] = {}
    for scene_name in scene_names:
        scene_payload_path = _build_scene_payload_path(
            results_root=results_root,
            scene_name=scene_name,
        )
        if force or not scene_payload_path.is_file():
            scene_payload = _build_scene_result(
                scene_name=scene_name,
                processed_gso_root=processed_gso_root,
                raw_gso_root=raw_gso_root,
            )
            scene_payload_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(scene_payload, scene_payload_path)
            _save_scene_summary(
                scene_payload=scene_payload,
                results_root=results_root,
            )
        else:
            scene_payload = torch.load(scene_payload_path, map_location="cpu")
        scene_summaries[scene_name] = _build_scene_summary_dict(
            scene_payload=scene_payload,
        )

    aggregate_summary = _build_aggregate_summary(
        scene_summaries=scene_summaries,
    )
    _save_aggregate_plot(
        aggregate_summary=aggregate_summary,
        results_root=results_root,
    )
    results_index = {
        "scene_names": scene_names,
        "default_scene_name": DEFAULT_SCENE_NAME,
        "open3d_gpu_supported": _open3d_gpu_device() is not None,
        "aggregate_summary": aggregate_summary,
    }
    index_path = results_root / "benchmark_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(results_index, handle, indent=2)
    return results_index


def load_results_index(
    results_root: Path,
) -> Dict[str, Any]:
    """Load the cached results index for the benchmark viewer.

    Args:
        results_root: Benchmark results root.

    Returns:
        Results-index dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )

    _validate_inputs()

    index_path = results_root / "benchmark_index.json"
    assert index_path.is_file(), (
        "Expected the benchmark index file to exist. " f"{index_path=}"
    )
    with index_path.open("r", encoding="utf-8") as handle:
        results_index = json.load(handle)
    assert isinstance(results_index, dict), (
        "Expected `results_index` to be a dict. " f"{type(results_index)=}."
    )
    return results_index


def load_scene_payload(
    scene_name: str,
    results_root: Path,
) -> Dict[str, Any]:
    """Load one cached scene payload for the benchmark viewer.

    Args:
        scene_name: Benchmark scene name.
        results_root: Benchmark results root.

    Returns:
        Scene payload dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )
        assert scene_name != "", (
            "Expected `scene_name` to be non-empty. " f"{scene_name=}"
        )
        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )

    _validate_inputs()

    scene_payload_path = _build_scene_payload_path(
        results_root=results_root,
        scene_name=scene_name,
    )
    assert scene_payload_path.is_file(), (
        "Expected the cached scene payload to exist. " f"{scene_payload_path=}"
    )
    scene_payload = torch.load(scene_payload_path, map_location="cpu")
    assert isinstance(scene_payload, dict), (
        "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
    )
    return scene_payload


def build_scene_timing_figure(
    scene_payload: Dict[str, Any],
) -> go.Figure:
    """Build the per-scene timing comparison figure for the viewer.

    Args:
        scene_payload: Cached scene payload dictionary.

    Returns:
        Plotly bar figure.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )
        assert "methods" in scene_payload, (
            "Expected `scene_payload` to contain `methods`. " f"{scene_payload.keys()=}"
        )

    _validate_inputs()

    ordered_method_keys = _list_method_keys(scene_payload=scene_payload)
    method_labels: List[str] = []
    visibility_means: List[float] = []
    other_means: List[float] = []
    total_means: List[float] = []
    total_stds: List[float] = []
    visibility_labels: List[str] = []
    other_labels: List[str] = []
    total_labels: List[str] = []
    for method_key in ordered_method_keys:
        method_payload = scene_payload["methods"][method_key]
        summary = method_payload["timing_summary"]
        ratio = summary["relative_to_v1_total_mean"]
        method_labels.append(f"{method_payload['display_label']}<br>{ratio:.2f}x")
        visibility_mean = summary["visibility_mean_ms"]
        other_mean = summary["other_mean_ms"]
        total_mean = summary["total_mean_ms"]
        total_std = summary["total_std_ms"]
        visibility_means.append(visibility_mean)
        other_means.append(other_mean)
        total_means.append(total_mean)
        total_stds.append(total_std)
        visibility_labels.append(
            ""
            if visibility_mean <= 0.0
            else f"{visibility_mean:.1f}±{summary['visibility_std_ms']:.1f}"
        )
        other_labels.append(
            ""
            if other_mean <= 0.0
            else f"{other_mean:.1f}±{summary['other_std_ms']:.1f}"
        )
        total_labels.append(f"{total_mean:.1f}±{total_std:.1f} ms")

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=method_labels,
            y=visibility_means,
            name="Visibility",
            marker_color="#1f77b4",
            text=visibility_labels,
            textposition="inside",
            hovertemplate="%{x}<br>Visibility: %{y:.2f} ms<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=method_labels,
            y=other_means,
            name="Other",
            marker_color="#ff7f0e",
            text=other_labels,
            textposition="inside",
            hovertemplate="%{x}<br>Other: %{y:.2f} ms<extra></extra>",
        )
    )
    for method_idx, total_label in enumerate(total_labels):
        figure.add_annotation(
            x=method_labels[method_idx],
            y=total_means[method_idx],
            text=total_label,
            showarrow=False,
            yshift=14,
            font={"size": 12, "color": "#0f172a"},
        )

    figure.update_layout(
        barmode="stack",
        title="Extraction Timing by Method",
        title_x=0.5,
        xaxis_title="Method and Ratio vs V1",
        yaxis_title="Time (ms)",
        legend_title="Breakdown",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 60, "r": 20, "t": 70, "b": 80},
    )
    figure.update_yaxes(gridcolor="#dbe4ee")
    return figure


def _build_scene_payload_path(
    results_root: Path,
    scene_name: str,
) -> Path:
    """Build the on-disk payload path for one scene.

    Args:
        results_root: Benchmark results root.
        scene_name: Benchmark scene name.

    Returns:
        Payload path.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )
        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )

    _validate_inputs()

    return results_root / scene_name / "scene_payload.pt"


def _build_scene_result(
    scene_name: str,
    processed_gso_root: Path,
    raw_gso_root: Path,
) -> Dict[str, Any]:
    """Build one full benchmark result payload for one `gso-clod` scene.

    Args:
        scene_name: Benchmark scene name.
        processed_gso_root: Root directory that contains processed `gso-clod` scenes.
        raw_gso_root: Root directory that contains raw GSO mesh assets.

    Returns:
        Scene payload dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )
        assert isinstance(processed_gso_root, Path), (
            "Expected `processed_gso_root` to be a `Path`. "
            f"{type(processed_gso_root)=}."
        )
        assert isinstance(raw_gso_root, Path), (
            "Expected `raw_gso_root` to be a `Path`. " f"{type(raw_gso_root)=}."
        )

    _validate_inputs()

    scene_context = _build_scene_context(
        scene_name=scene_name,
        processed_gso_root=processed_gso_root,
        raw_gso_root=raw_gso_root,
    )
    cuda_inputs = _build_cuda_extraction_inputs(
        scene_context=scene_context,
    )
    try:
        texel_visibility_v1 = _benchmark_texel_visibility_method(
            scene_name=scene_name,
            cuda_inputs=cuda_inputs,
            use_v2=False,
        )
        texel_visibility_v2 = _benchmark_texel_visibility_method(
            scene_name=scene_name,
            cuda_inputs=cuda_inputs,
            use_v2=True,
        )
        open3d_cpu = _benchmark_open3d_cpu_method(
            scene_context=scene_context,
            uv_occupancy_mask=cuda_inputs["uv_occupancy_mask_cpu"],
        )
    finally:
        del cuda_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    methods = {
        "texel_visibility_v1": texel_visibility_v1,
        "texel_visibility_v2": texel_visibility_v2,
        "open3d_cpu": open3d_cpu,
    }
    if _open3d_gpu_device() is not None:
        methods["open3d_gpu"] = {
            "display_label": "Open3D GPU",
            "timing_summary": {
                "visibility_mean_ms": 0.0,
                "visibility_std_ms": 0.0,
                "other_mean_ms": 0.0,
                "other_std_ms": 0.0,
                "total_mean_ms": 0.0,
                "total_std_ms": 0.0,
                "relative_to_v1_total_mean": 0.0,
            },
            "timings_ms": {"visibility": [], "other": [], "total": []},
            "uv_texture_map": torch.zeros_like(open3d_cpu["uv_texture_map"]),
            "supported": False,
        }

    baseline_total_mean = methods["texel_visibility_v1"]["timing_summary"][
        "total_mean_ms"
    ]
    for method_key, method_payload in methods.items():
        total_mean = method_payload["timing_summary"]["total_mean_ms"]
        ratio = total_mean / baseline_total_mean if baseline_total_mean > 0.0 else 0.0
        method_payload["timing_summary"]["relative_to_v1_total_mean"] = ratio

    scene_payload = {
        "scene_name": scene_name,
        "source_rgb": scene_context["source_rgb"],
        "reference_texture_rgb": scene_context["reference_texture_rgb"],
        "mesh_vertices": scene_context["mesh_vertices"].detach().cpu(),
        "mesh_faces": scene_context["mesh_faces"].detach().cpu(),
        "mesh_vertex_uv": scene_context["mesh_vertex_uv"].detach().cpu(),
        "mesh_face_uvs": scene_context["mesh_face_uvs"].detach().cpu(),
        "methods": {
            method_key: {
                **method_payload,
                "uv_texture_map": method_payload["uv_texture_map"].detach().cpu(),
            }
            for method_key, method_payload in methods.items()
            if method_payload.get("supported", True)
        },
    }
    return scene_payload


def _build_scene_context(
    scene_name: str,
    processed_gso_root: Path,
    raw_gso_root: Path,
) -> Dict[str, Any]:
    """Build one reusable CPU scene context for benchmarking.

    Args:
        scene_name: Benchmark scene name.
        processed_gso_root: Root directory that contains processed `gso-clod` scenes.
        raw_gso_root: Root directory that contains raw GSO mesh assets.

    Returns:
        Scene-context dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )
        assert isinstance(processed_gso_root, Path), (
            "Expected `processed_gso_root` to be a `Path`. "
            f"{type(processed_gso_root)=}."
        )
        assert isinstance(raw_gso_root, Path), (
            "Expected `raw_gso_root` to be a `Path`. " f"{type(raw_gso_root)=}."
        )

    _validate_inputs()

    processed_scene_root = processed_gso_root / scene_name
    raw_scene_root = raw_gso_root / scene_name
    assert (processed_scene_root / "transforms.json").is_file(), (
        "Expected the processed scene transform file to exist. "
        f"{processed_scene_root=}"
    )
    assert (processed_scene_root / "sparse_pc.ply").is_file(), (
        "Expected the processed scene sparse point cloud to exist. "
        f"{processed_scene_root=}"
    )
    assert (raw_scene_root / "meshes" / "model.obj").is_file(), (
        "Expected the raw scene OBJ mesh to exist. " f"{raw_scene_root=}"
    )

    transforms_data = NerfStudio_Data.load(
        filepath=processed_scene_root / "transforms.json",
        device="cpu",
    )
    point_cloud = load_point_cloud(
        filepath=processed_scene_root / "sparse_pc.ply",
        device="cpu",
    )
    assert isinstance(point_cloud, PointCloud), (
        "Expected `point_cloud` to be a `PointCloud`. " f"{type(point_cloud)=}."
    )
    mesh = Mesh.load(path=raw_scene_root / "meshes" / "model.obj")
    assert mesh.texture_mode == "uv_texture_map", (
        "Expected the benchmark mesh to be UV textured. "
        f"{mesh.texture_mode=} {raw_scene_root=}"
    )
    assert mesh.vertex_uv is not None, (
        "Expected the benchmark mesh to contain UV coordinates. " f"{mesh.vertex_uv=}"
    )
    assert mesh.face_uvs is not None, (
        "Expected the benchmark mesh to contain UV-face indices. " f"{mesh.face_uvs=}"
    )
    normalized_mesh_vertices = _normalize_gso_mesh_vertices_to_processed_frame(
        mesh_vertices=mesh.vertices.to(dtype=torch.float32),
        point_cloud=point_cloud,
    )
    frames = transforms_data.data["frames"]
    assert isinstance(frames, list), (
        "Expected NerfStudio `frames` to be a list. " f"{type(frames)=}."
    )
    assert len(frames) > 0, "Expected at least one frame in `transforms.json`."
    source_image_path = processed_scene_root / frames[0]["file_path"]
    source_rgb = _load_rgb_image(image_path=source_image_path)
    reference_texture_rgb = _load_rgb_image(
        image_path=_resolve_reference_texture_path(raw_scene_root=raw_scene_root),
    )
    image_chw = (
        torch.from_numpy(source_rgb)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div(255.0)
        .contiguous()
    )
    exploded_geometry = _build_exploded_uv_geometry(
        mesh_vertices=normalized_mesh_vertices,
        mesh_faces=mesh.faces.to(dtype=torch.long),
        mesh_vertex_uv=mesh.vertex_uv.to(dtype=torch.float32),
        mesh_face_uvs=mesh.face_uvs.to(dtype=torch.long),
    )
    return {
        "scene_name": scene_name,
        "camera": transforms_data.cameras[0:1].to(device="cpu", convention="opencv"),
        "source_rgb": source_rgb,
        "reference_texture_rgb": reference_texture_rgb,
        "image_chw": image_chw,
        "mesh_vertices": normalized_mesh_vertices.contiguous(),
        "mesh_faces": mesh_faces.verts_idx.to(dtype=torch.long).contiguous(),
        "mesh_vertex_uv": mesh_aux.verts_uvs.to(dtype=torch.float32).contiguous(),
        "mesh_face_uvs": mesh_faces.textures_idx.to(dtype=torch.long).contiguous(),
        "exploded_vertices": exploded_geometry["vertices"],
        "exploded_faces": exploded_geometry["faces"],
        "exploded_vertex_uv": exploded_geometry["vertex_uv"],
        "texture_size": int(source_rgb.shape[0]),
    }


def _resolve_reference_texture_path(
    raw_scene_root: Path,
) -> Path:
    """Resolve the real reference texture path for one raw GSO asset.

    Args:
        raw_scene_root: Raw GSO scene root.

    Returns:
        Reference texture path.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(raw_scene_root, Path), (
            "Expected `raw_scene_root` to be a `Path`. " f"{type(raw_scene_root)=}."
        )
        assert (raw_scene_root / "meshes" / "model.mtl").is_file(), (
            "Expected the raw GSO material file to exist. " f"{raw_scene_root=}"
        )

    _validate_inputs()

    mtl_path = raw_scene_root / "meshes" / "model.mtl"
    referenced_filename: Optional[str] = None
    for line in mtl_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("map_Kd "):
            referenced_filename = stripped_line.split(maxsplit=1)[1]
            break
    assert referenced_filename is not None, (
        "Expected `model.mtl` to contain one `map_Kd` entry. " f"{mtl_path=}"
    )
    candidate_paths = [
        raw_scene_root / "meshes" / referenced_filename,
        raw_scene_root / "materials" / "textures" / referenced_filename,
    ]
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path
    assert False, (
        "Failed to resolve the raw reference texture path from `model.mtl`. "
        f"{candidate_paths=}"
    )


def _normalize_gso_mesh_vertices_to_processed_frame(
    mesh_vertices: torch.Tensor,
    point_cloud: PointCloud,
) -> torch.Tensor:
    """Normalize one raw GSO mesh into the processed scene frame.

    Args:
        mesh_vertices: Raw mesh vertices `[V, 3]`.
        point_cloud: Processed-scene sparse point cloud.

    Returns:
        Mesh vertices aligned to the processed scene frame.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(mesh_vertices, torch.Tensor), (
            "Expected `mesh_vertices` to be a tensor. " f"{type(mesh_vertices)=}."
        )
        assert isinstance(point_cloud, PointCloud), (
            "Expected `point_cloud` to be a `PointCloud`. " f"{type(point_cloud)=}."
        )

    _validate_inputs()

    mesh_min = mesh_vertices.min(dim=0).values
    mesh_max = mesh_vertices.max(dim=0).values
    mesh_center = 0.5 * (mesh_min + mesh_max)
    mesh_half_extent = 0.5 * (mesh_max - mesh_min)
    point_xyz = point_cloud.xyz.detach().to(dtype=torch.float32, device="cpu")
    point_min = point_xyz.min(dim=0).values
    point_max = point_xyz.max(dim=0).values
    point_center = 0.5 * (point_min + point_max)
    point_half_extent = 0.5 * (point_max - point_min)
    mesh_scale = torch.clamp(mesh_half_extent.max(), min=1e-8)
    point_scale = torch.clamp(point_half_extent.max(), min=1e-8)
    return (
        (mesh_vertices - mesh_center) * (point_scale / mesh_scale) + point_center
    ).contiguous()


def _build_exploded_uv_geometry(
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    mesh_vertex_uv: torch.Tensor,
    mesh_face_uvs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build one exploded UV mesh for single-material texture extraction.

    Args:
        mesh_vertices: Original mesh vertices `[V, 3]`.
        mesh_faces: Original mesh faces `[F, 3]`.
        mesh_vertex_uv: Original mesh UV table `[U, 2]`.
        mesh_face_uvs: Original face-to-UV indices `[F, 3]`.

    Returns:
        Dict containing exploded vertices, faces, and UV coordinates.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(mesh_vertices, torch.Tensor), (
            "Expected `mesh_vertices` to be a tensor. " f"{type(mesh_vertices)=}."
        )
        assert isinstance(mesh_faces, torch.Tensor), (
            "Expected `mesh_faces` to be a tensor. " f"{type(mesh_faces)=}."
        )
        assert isinstance(mesh_vertex_uv, torch.Tensor), (
            "Expected `mesh_vertex_uv` to be a tensor. " f"{type(mesh_vertex_uv)=}."
        )
        assert isinstance(mesh_face_uvs, torch.Tensor), (
            "Expected `mesh_face_uvs` to be a tensor. " f"{type(mesh_face_uvs)=}."
        )

    _validate_inputs()

    face_vertices = mesh_vertices[mesh_faces].contiguous()
    face_vertex_uv = mesh_vertex_uv[mesh_face_uvs].contiguous()
    edge01 = face_vertices[:, 1] - face_vertices[:, 0]
    edge02 = face_vertices[:, 2] - face_vertices[:, 0]
    face_area_norm = torch.linalg.norm(
        torch.cross(edge01, edge02, dim=1),
        dim=1,
    )
    valid_face_mask = face_area_norm > 0.0
    assert torch.any(valid_face_mask), (
        "Expected at least one non-degenerate face in the exploded extraction mesh. "
        f"{valid_face_mask.shape=}."
    )
    filtered_face_vertices = face_vertices[valid_face_mask].contiguous()
    filtered_face_vertex_uv = face_vertex_uv[valid_face_mask].contiguous()
    exploded_vertices = filtered_face_vertices.reshape(-1, 3).contiguous()
    exploded_vertex_uv = filtered_face_vertex_uv.reshape(-1, 2).contiguous()
    exploded_faces = torch.arange(
        exploded_vertices.shape[0],
        dtype=torch.long,
    ).reshape(-1, 3)
    return {
        "vertices": exploded_vertices,
        "faces": exploded_faces,
        "vertex_uv": exploded_vertex_uv,
    }


def _build_cuda_extraction_inputs(
    scene_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build reusable CUDA extraction inputs for the v1/v2 methods.

    Args:
        scene_context: CPU scene-context dictionary.

    Returns:
        CUDA extraction-input dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_context, dict), (
            "Expected `scene_context` to be a dict. " f"{type(scene_context)=}."
        )
        assert torch.cuda.is_available(), (
            "Expected CUDA to be available for the v1/v2 benchmark methods. "
            f"{torch.cuda.is_available()=}"
        )

    _validate_inputs()

    device = _select_cuda_device()
    torch.cuda.set_device(device)
    exploded_vertices_cuda = scene_context["exploded_vertices"].to(
        device=device,
        dtype=torch.float32,
    )
    exploded_faces_cuda = scene_context["exploded_faces"].to(
        device=device,
        dtype=torch.long,
    )
    exploded_vertex_uv_cuda = scene_context["exploded_vertex_uv"].to(
        device=device,
        dtype=torch.float32,
    )
    image_cuda = scene_context["image_chw"].to(
        device=device,
        dtype=torch.float32,
    )
    camera_cuda = scene_context["camera"].to(device=device, convention="opencv")
    uv_rasterization_data = _build_uv_rasterization_data(
        vertices=exploded_vertices_cuda,
        vertex_uv=exploded_vertex_uv_cuda,
        faces=exploded_faces_cuda,
        texture_size=scene_context["texture_size"],
    )
    return {
        "device": device,
        "vertices": exploded_vertices_cuda,
        "faces": exploded_faces_cuda,
        "vertex_uv": exploded_vertex_uv_cuda,
        "image": image_cuda,
        "camera": camera_cuda,
        "uv_rasterization_data": uv_rasterization_data,
        "uv_occupancy_mask_cpu": uv_rasterization_data["uv_mask"].detach().cpu(),
        "weights_cfg": normalize_weights_cfg(
            weights_cfg=DEFAULT_WEIGHTS_CFG,
            default_weights="visible",
        ),
    }


def _benchmark_texel_visibility_method(
    scene_name: str,
    cuda_inputs: Dict[str, Any],
    use_v2: bool,
) -> Dict[str, Any]:
    """Benchmark one CUDA texel-visibility extraction method.

    Args:
        scene_name: Benchmark scene name.
        cuda_inputs: Reusable CUDA extraction-input dictionary.
        use_v2: Whether to benchmark the v2 visibility method.

    Returns:
        Method-result dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )
        assert isinstance(cuda_inputs, dict), (
            "Expected `cuda_inputs` to be a dict. " f"{type(cuda_inputs)=}."
        )
        assert isinstance(use_v2, bool), (
            "Expected `use_v2` to be a bool. " f"{type(use_v2)=}."
        )

    _validate_inputs()

    texture_output: Optional[torch.Tensor] = None
    visibility_timings_ms: List[float] = []
    other_timings_ms: List[float] = []
    total_timings_ms: List[float] = []
    for repeat_idx in range(WARMUP_RUNS + REPETITIONS):
        run_output = _run_single_texel_visibility_extraction(
            cuda_inputs=cuda_inputs,
            use_v2=use_v2,
        )
        if repeat_idx >= WARMUP_RUNS:
            texture_output = run_output["uv_texture_map"].detach().cpu()
            visibility_timings_ms.append(run_output["visibility_ms"])
            other_timings_ms.append(run_output["other_ms"])
            total_timings_ms.append(
                run_output["visibility_ms"] + run_output["other_ms"]
            )

    assert texture_output is not None, (
        "Expected the benchmarked texture output to be populated. " f"{texture_output=}"
    )
    return {
        "display_label": "Texel Visibility V2" if use_v2 else "Texel Visibility V1",
        "timings_ms": {
            "visibility": visibility_timings_ms,
            "other": other_timings_ms,
            "total": total_timings_ms,
        },
        "timing_summary": _build_timing_summary(
            visibility_timings_ms=visibility_timings_ms,
            other_timings_ms=other_timings_ms,
            total_timings_ms=total_timings_ms,
        ),
        "uv_texture_map": _normalize_texture_tensor_for_payload(
            uv_texture_map=texture_output
        ),
        "supported": True,
        "scene_name": scene_name,
    }


def _run_single_texel_visibility_extraction(
    cuda_inputs: Dict[str, Any],
    use_v2: bool,
) -> Dict[str, Any]:
    """Run one timed CUDA texture extraction with v1 or v2 visibility.

    Args:
        cuda_inputs: Reusable CUDA extraction-input dictionary.
        use_v2: Whether to use the v2 visibility method.

    Returns:
        Dict containing the extracted texture map and measured timings.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(cuda_inputs, dict), (
            "Expected `cuda_inputs` to be a dict. " f"{type(cuda_inputs)=}."
        )
        assert isinstance(use_v2, bool), (
            "Expected `use_v2` to be a bool. " f"{type(use_v2)=}."
        )

    _validate_inputs()

    vertices = cuda_inputs["vertices"]
    faces = cuda_inputs["faces"]
    image = cuda_inputs["image"]
    camera = cuda_inputs["camera"]
    uv_rasterization_data = cuda_inputs["uv_rasterization_data"]
    weights_cfg = cuda_inputs["weights_cfg"]
    cuda_device = cuda_inputs["device"]
    torch.cuda.synchronize(device=cuda_device)
    visibility_start_time = time.perf_counter()
    uv_visibility_mask = (
        compute_f_visibility_mask_v2(
            vertices=vertices,
            faces=faces,
            camera=camera,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
            uv_rasterization_data=uv_rasterization_data,
        )
        if use_v2
        else compute_f_visibility_mask(
            vertices=vertices,
            faces=faces,
            camera=camera,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
            uv_rasterization_data=uv_rasterization_data,
        )
    )
    torch.cuda.synchronize(device=cuda_device)
    visibility_ms = (time.perf_counter() - visibility_start_time) * 1000.0
    torch.cuda.synchronize(device=cuda_device)
    other_start_time = time.perf_counter()
    face_normals_weight = _compute_f_normals_weights(
        vertices=vertices,
        faces=faces,
        camera=camera,
        weights_cfg=weights_cfg,
    )
    uv_normals_weight = _rasterize_face_weights_to_uv(
        face_weight=face_normals_weight,
        uv_rasterization_data=uv_rasterization_data,
    )
    uv_weight = uv_visibility_mask * uv_normals_weight
    uv_texture = _project_f_colors(
        vertices=vertices,
        image=image,
        camera=camera,
        uv_rasterization_data=uv_rasterization_data,
    )
    uv_valid_mask = (uv_weight > 0.0).to(dtype=torch.float32)
    uv_texture_map = _apply_invisible_texel_noise(
        uv_texture_map=uv_texture,
        uv_valid_mask=uv_valid_mask,
        uv_occupancy_mask=uv_rasterization_data["uv_mask"],
    )
    torch.cuda.synchronize(device=cuda_device)
    other_ms = (time.perf_counter() - other_start_time) * 1000.0
    return {
        "visibility_ms": visibility_ms,
        "other_ms": other_ms,
        "uv_texture_map": uv_texture_map,
    }


def _benchmark_open3d_cpu_method(
    scene_context: Dict[str, Any],
    uv_occupancy_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Benchmark the CPU Open3D baseline method.

    Args:
        scene_context: CPU scene-context dictionary.
        uv_occupancy_mask: CPU UV occupancy mask `[1, T, T, 1]`.

    Returns:
        Method-result dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_context, dict), (
            "Expected `scene_context` to be a dict. " f"{type(scene_context)=}."
        )
        assert isinstance(uv_occupancy_mask, torch.Tensor), (
            "Expected `uv_occupancy_mask` to be a tensor. "
            f"{type(uv_occupancy_mask)=}."
        )

    _validate_inputs()

    source_rgb_float = scene_context["source_rgb"].astype(np.float32) / 255.0
    camera = scene_context["camera"][0]
    intrinsic_matrix = o3d.core.Tensor(
        camera.intrinsics.detach().cpu().numpy().astype(np.float32)
    )
    extrinsic_matrix = o3d.core.Tensor(
        camera.extrinsics.detach().cpu().numpy().astype(np.float32)
    )
    triangle_uvs = (
        scene_context["exploded_vertex_uv"]
        .reshape(-1, 3, 2)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(
            scene_context["exploded_vertices"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            dtype=o3d.core.Dtype.Float32,
        ),
        o3d.core.Tensor(
            scene_context["exploded_faces"].detach().cpu().numpy().astype(np.uint32),
            dtype=o3d.core.Dtype.UInt32,
        ),
    )
    texture_output: Optional[torch.Tensor] = None
    total_timings_ms: List[float] = []
    for repeat_idx in range(WARMUP_RUNS + REPETITIONS):
        run_output = _run_single_open3d_cpu_extraction(
            raycasting_scene=scene,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            width_px=int(scene_context["source_rgb"].shape[1]),
            height_px=int(scene_context["source_rgb"].shape[0]),
            source_rgb_float=source_rgb_float,
            triangle_uvs=triangle_uvs,
            texture_size=scene_context["texture_size"],
            uv_occupancy_mask=uv_occupancy_mask,
        )
        if repeat_idx >= WARMUP_RUNS:
            texture_output = run_output["uv_texture_map"].detach().cpu()
            total_timings_ms.append(run_output["total_ms"])
    assert texture_output is not None, (
        "Expected the Open3D texture output to be populated. " f"{texture_output=}"
    )
    return {
        "display_label": "Open3D CPU",
        "timings_ms": {
            "visibility": [0.0 for _ in total_timings_ms],
            "other": list(total_timings_ms),
            "total": list(total_timings_ms),
        },
        "timing_summary": _build_timing_summary(
            visibility_timings_ms=[0.0 for _ in total_timings_ms],
            other_timings_ms=total_timings_ms,
            total_timings_ms=total_timings_ms,
        ),
        "uv_texture_map": _normalize_texture_tensor_for_payload(
            uv_texture_map=texture_output
        ),
        "supported": True,
    }


def _run_single_open3d_cpu_extraction(
    raycasting_scene: o3d.t.geometry.RaycastingScene,
    intrinsic_matrix: o3d.core.Tensor,
    extrinsic_matrix: o3d.core.Tensor,
    width_px: int,
    height_px: int,
    source_rgb_float: np.ndarray,
    triangle_uvs: np.ndarray,
    texture_size: int,
    uv_occupancy_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Run one timed Open3D CPU extraction.

    Args:
        raycasting_scene: Open3D raycasting scene.
        intrinsic_matrix: Camera intrinsics `(3, 3)`.
        extrinsic_matrix: Camera extrinsics `(4, 4)` in world-to-camera form.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        source_rgb_float: Source RGB image `[H, W, 3]` in `[0, 1]`.
        triangle_uvs: Per-triangle UV coordinates `[F, 3, 2]`.
        texture_size: Square UV texture size.
        uv_occupancy_mask: UV occupancy mask `[1, T, T, 1]`.

    Returns:
        Dict containing the extracted texture map and total timing.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(raycasting_scene, o3d.t.geometry.RaycastingScene), (
            "Expected `raycasting_scene` to be an Open3D `RaycastingScene`. "
            f"{type(raycasting_scene)=}."
        )
        assert isinstance(intrinsic_matrix, o3d.core.Tensor), (
            "Expected `intrinsic_matrix` to be an Open3D tensor. "
            f"{type(intrinsic_matrix)=}."
        )
        assert isinstance(extrinsic_matrix, o3d.core.Tensor), (
            "Expected `extrinsic_matrix` to be an Open3D tensor. "
            f"{type(extrinsic_matrix)=}."
        )
        assert isinstance(width_px, int), (
            "Expected `width_px` to be an int. " f"{type(width_px)=}."
        )
        assert isinstance(height_px, int), (
            "Expected `height_px` to be an int. " f"{type(height_px)=}."
        )
        assert isinstance(source_rgb_float, np.ndarray), (
            "Expected `source_rgb_float` to be a numpy array. "
            f"{type(source_rgb_float)=}."
        )
        assert isinstance(triangle_uvs, np.ndarray), (
            "Expected `triangle_uvs` to be a numpy array. " f"{type(triangle_uvs)=}."
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an int. " f"{type(texture_size)=}."
        )
        assert isinstance(uv_occupancy_mask, torch.Tensor), (
            "Expected `uv_occupancy_mask` to be a tensor. "
            f"{type(uv_occupancy_mask)=}."
        )

    _validate_inputs()

    start_time = time.perf_counter()
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
        width_px=width_px,
        height_px=height_px,
    )
    raycast_output = raycasting_scene.cast_rays(rays)
    hit_mask = np.isfinite(raycast_output["t_hit"].numpy())
    primitive_ids = raycast_output["primitive_ids"].numpy()
    primitive_uvs = raycast_output["primitive_uvs"].numpy()
    uv_texture_map = torch.zeros(
        (1, texture_size, texture_size, 3),
        dtype=torch.float32,
    )
    uv_valid_mask = torch.zeros(
        (1, texture_size, texture_size, 1),
        dtype=torch.float32,
    )
    if np.any(hit_mask):
        hit_triangle_ids = primitive_ids[hit_mask].astype(np.int64)
        hit_triangle_uvs = triangle_uvs[hit_triangle_ids]
        hit_primitive_uvs = primitive_uvs[hit_mask].astype(np.float32)
        barycentric_coords = np.stack(
            [
                1.0 - hit_primitive_uvs[:, 0] - hit_primitive_uvs[:, 1],
                hit_primitive_uvs[:, 0],
                hit_primitive_uvs[:, 1],
            ],
            axis=1,
        ).astype(np.float32)
        hit_uv = np.sum(
            hit_triangle_uvs * barycentric_coords[:, :, None],
            axis=1,
        ).astype(np.float32)
        hit_uv[:, 0] = np.mod(hit_uv[:, 0], 1.0)
        hit_uv[:, 1] = np.clip(hit_uv[:, 1], a_min=0.0, a_max=1.0 - 1e-6)
        texel_x = np.clip(
            (hit_uv[:, 0] * float(texture_size)).astype(np.int64),
            a_min=0,
            a_max=texture_size - 1,
        )
        texel_y = np.clip(
            (hit_uv[:, 1] * float(texture_size)).astype(np.int64),
            a_min=0,
            a_max=texture_size - 1,
        )
        linear_index = texel_y * texture_size + texel_x
        source_colors = source_rgb_float[hit_mask].astype(np.float32)
        texture_sum = np.zeros((texture_size * texture_size, 3), dtype=np.float32)
        texture_count = np.zeros((texture_size * texture_size, 1), dtype=np.float32)
        np.add.at(texture_sum, linear_index, source_colors)
        np.add.at(texture_count[:, 0], linear_index, 1.0)
        valid_texel_mask = texture_count[:, 0] > 0.0
        texture_mean = np.zeros_like(texture_sum)
        texture_mean[valid_texel_mask] = (
            texture_sum[valid_texel_mask] / texture_count[valid_texel_mask]
        )
        uv_texture_map = torch.from_numpy(
            texture_mean.reshape(1, texture_size, texture_size, 3)
        ).to(dtype=torch.float32)
        uv_valid_mask = torch.from_numpy(
            valid_texel_mask.reshape(1, texture_size, texture_size, 1).astype(
                np.float32
            )
        )
    uv_texture_map = _apply_invisible_texel_noise(
        uv_texture_map=uv_texture_map,
        uv_valid_mask=uv_valid_mask,
        uv_occupancy_mask=uv_occupancy_mask,
    )
    total_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "total_ms": total_ms,
        "uv_texture_map": uv_texture_map,
    }


def _build_deterministic_invisible_texel_noise(
    texture_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic seed-zero invisible-texel noise.

    Args:
        texture_size: Square UV texture size.
        device: Target device.

    Returns:
        Noise tensor `[1, T, T, 3]` in `[0, 1]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an int. " f"{type(texture_size)=}."
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )
        assert isinstance(device, torch.device), (
            "Expected `device` to be a `torch.device`. " f"{type(device)=}."
        )

    _validate_inputs()

    random_generator = np.random.default_rng(seed=0)
    noise = random_generator.random(
        size=(1, texture_size, texture_size, 3),
        dtype=np.float32,
    )
    return torch.from_numpy(noise).to(device=device)


def _apply_invisible_texel_noise(
    uv_texture_map: torch.Tensor,
    uv_valid_mask: torch.Tensor,
    uv_occupancy_mask: torch.Tensor,
) -> torch.Tensor:
    """Fill invisible occupied texels with deterministic seed-zero noise.

    Args:
        uv_texture_map: UV texture map `[1, T, T, 3]`.
        uv_valid_mask: UV valid-observation mask `[1, T, T, 1]`.
        uv_occupancy_mask: UV occupancy mask `[1, T, T, 1]`.

    Returns:
        Noise-filled UV texture map `[1, T, T, 3]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(uv_texture_map, torch.Tensor), (
            "Expected `uv_texture_map` to be a tensor. " f"{type(uv_texture_map)=}."
        )
        assert isinstance(uv_valid_mask, torch.Tensor), (
            "Expected `uv_valid_mask` to be a tensor. " f"{type(uv_valid_mask)=}."
        )
        assert isinstance(uv_occupancy_mask, torch.Tensor), (
            "Expected `uv_occupancy_mask` to be a tensor. "
            f"{type(uv_occupancy_mask)=}."
        )
        assert uv_texture_map.ndim == 4, (
            "Expected `uv_texture_map` to have shape `[1, T, T, 3]`. "
            f"{uv_texture_map.shape=}"
        )
        assert uv_valid_mask.shape[:3] == uv_texture_map.shape[:3], (
            "Expected `uv_valid_mask` to align with `uv_texture_map`. "
            f"{uv_valid_mask.shape=} {uv_texture_map.shape=}"
        )
        assert uv_occupancy_mask.shape[:3] == uv_texture_map.shape[:3], (
            "Expected `uv_occupancy_mask` to align with `uv_texture_map`. "
            f"{uv_occupancy_mask.shape=} {uv_texture_map.shape=}"
        )

    _validate_inputs()

    noise = _build_deterministic_invisible_texel_noise(
        texture_size=int(uv_texture_map.shape[1]),
        device=uv_texture_map.device,
    )
    invisible_mask = (uv_occupancy_mask > 0.5) & (uv_valid_mask <= 0.5)
    unoccupied_mask = uv_occupancy_mask <= 0.5
    uv_texture_with_noise = torch.where(
        invisible_mask.expand_as(uv_texture_map),
        noise,
        uv_texture_map,
    )
    return torch.where(
        unoccupied_mask.expand_as(uv_texture_with_noise),
        torch.zeros_like(uv_texture_with_noise),
        uv_texture_with_noise,
    ).contiguous()


def _normalize_texture_tensor_for_payload(
    uv_texture_map: torch.Tensor,
) -> torch.Tensor:
    """Normalize one UV texture tensor to CPU HWC float32 layout.

    Args:
        uv_texture_map: UV texture tensor in NHWC or HWC format.

    Returns:
        CPU HWC float32 UV texture tensor.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(uv_texture_map, torch.Tensor), (
            "Expected `uv_texture_map` to be a tensor. " f"{type(uv_texture_map)=}."
        )

    _validate_inputs()

    normalized_texture = uv_texture_map.detach().to(dtype=torch.float32)
    if normalized_texture.ndim == 4:
        assert normalized_texture.shape[0] == 1, (
            "Expected batched UV payload normalization to receive batch size 1. "
            f"{normalized_texture.shape=}"
        )
        normalized_texture = normalized_texture[0]
    assert normalized_texture.ndim == 3, (
        "Expected normalized UV payload texture to be rank 3. "
        f"{normalized_texture.shape=}"
    )
    if normalized_texture.shape[0] == 3:
        normalized_texture = normalized_texture.permute(1, 2, 0).contiguous()
    assert normalized_texture.shape[2] == 3, (
        "Expected normalized UV payload texture to end in RGB channels. "
        f"{normalized_texture.shape=}"
    )
    return normalized_texture.detach().cpu().clamp(0.0, 1.0).contiguous()


def _build_timing_summary(
    visibility_timings_ms: List[float],
    other_timings_ms: List[float],
    total_timings_ms: List[float],
) -> Dict[str, float]:
    """Build one timing-summary dictionary from per-repeat measurements.

    Args:
        visibility_timings_ms: Visibility timings in milliseconds.
        other_timings_ms: Other-step timings in milliseconds.
        total_timings_ms: Total timings in milliseconds.

    Returns:
        Timing-summary dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(visibility_timings_ms, list), (
            "Expected `visibility_timings_ms` to be a list. "
            f"{type(visibility_timings_ms)=}."
        )
        assert isinstance(other_timings_ms, list), (
            "Expected `other_timings_ms` to be a list. " f"{type(other_timings_ms)=}."
        )
        assert isinstance(total_timings_ms, list), (
            "Expected `total_timings_ms` to be a list. " f"{type(total_timings_ms)=}."
        )
        assert len(visibility_timings_ms) == len(other_timings_ms), (
            "Expected visibility and other timings to share a length. "
            f"{len(visibility_timings_ms)=} {len(other_timings_ms)=}."
        )
        assert len(visibility_timings_ms) == len(total_timings_ms), (
            "Expected visibility and total timings to share a length. "
            f"{len(visibility_timings_ms)=} {len(total_timings_ms)=}."
        )
        assert len(total_timings_ms) == REPETITIONS, (
            "Expected each timing summary to cover `REPETITIONS` runs. "
            f"{len(total_timings_ms)=} {REPETITIONS=}"
        )

    _validate_inputs()

    visibility_array = np.asarray(visibility_timings_ms, dtype=np.float64)
    other_array = np.asarray(other_timings_ms, dtype=np.float64)
    total_array = np.asarray(total_timings_ms, dtype=np.float64)
    return {
        "visibility_mean_ms": float(visibility_array.mean()),
        "visibility_std_ms": float(visibility_array.std(ddof=0)),
        "other_mean_ms": float(other_array.mean()),
        "other_std_ms": float(other_array.std(ddof=0)),
        "total_mean_ms": float(total_array.mean()),
        "total_std_ms": float(total_array.std(ddof=0)),
        "relative_to_v1_total_mean": 0.0,
    }


def _save_scene_summary(
    scene_payload: Dict[str, Any],
    results_root: Path,
) -> None:
    """Save one JSON timing summary next to the cached scene payload.

    Args:
        scene_payload: Cached scene payload dictionary.
        results_root: Benchmark results root.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )
        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )

    _validate_inputs()

    summary_path = results_root / scene_payload["scene_name"] / "timing_summary.json"
    summary_dict = _build_scene_summary_dict(scene_payload=scene_payload)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_dict, handle, indent=2)


def _build_scene_summary_dict(
    scene_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the JSON-serializable timing summary for one scene payload.

    Args:
        scene_payload: Cached scene payload dictionary.

    Returns:
        Scene-summary dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )

    _validate_inputs()

    return {
        "scene_name": scene_payload["scene_name"],
        "methods": {
            method_key: {
                "display_label": method_payload["display_label"],
                "timings_ms": method_payload["timings_ms"],
                "timing_summary": method_payload["timing_summary"],
            }
            for method_key, method_payload in scene_payload["methods"].items()
        },
    }


def _build_aggregate_summary(
    scene_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the aggregate summary averaged over all benchmark scenes.

    Args:
        scene_summaries: Mapping from scene name to scene-summary dictionary.

    Returns:
        Aggregate-summary dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_summaries, dict), (
            "Expected `scene_summaries` to be a dict. " f"{type(scene_summaries)=}."
        )
        assert len(scene_summaries) == 30, (
            "Expected the aggregate summary to cover 30 scenes. "
            f"{len(scene_summaries)=}."
        )

    _validate_inputs()

    method_keys = _list_method_keys_from_summary(
        scene_summary=next(iter(scene_summaries.values()))
    )
    aggregate_methods: Dict[str, Any] = {}
    baseline_means: List[float] = [
        scene_summaries[scene_name]["methods"]["texel_visibility_v1"]["timing_summary"][
            "total_mean_ms"
        ]
        for scene_name in sorted(scene_summaries.keys())
    ]
    baseline_mean = float(np.mean(np.asarray(baseline_means, dtype=np.float64)))
    for method_key in method_keys:
        visibility_means = []
        visibility_stds = []
        other_means = []
        other_stds = []
        total_means = []
        total_stds = []
        display_label = None
        for scene_name in sorted(scene_summaries.keys()):
            method_summary = scene_summaries[scene_name]["methods"][method_key]
            display_label = method_summary["display_label"]
            timing_summary = method_summary["timing_summary"]
            visibility_means.append(timing_summary["visibility_mean_ms"])
            visibility_stds.append(timing_summary["visibility_std_ms"])
            other_means.append(timing_summary["other_mean_ms"])
            other_stds.append(timing_summary["other_std_ms"])
            total_means.append(timing_summary["total_mean_ms"])
            total_stds.append(timing_summary["total_std_ms"])
        aggregate_total_mean = float(np.mean(np.asarray(total_means, dtype=np.float64)))
        aggregate_methods[method_key] = {
            "display_label": display_label,
            "visibility_mean_ms": float(
                np.mean(np.asarray(visibility_means, dtype=np.float64))
            ),
            "visibility_std_ms": float(
                np.mean(np.asarray(visibility_stds, dtype=np.float64))
            ),
            "other_mean_ms": float(np.mean(np.asarray(other_means, dtype=np.float64))),
            "other_std_ms": float(np.mean(np.asarray(other_stds, dtype=np.float64))),
            "total_mean_ms": aggregate_total_mean,
            "total_std_ms": float(np.mean(np.asarray(total_stds, dtype=np.float64))),
            "relative_to_v1_total_mean": (
                aggregate_total_mean / baseline_mean if baseline_mean > 0.0 else 0.0
            ),
        }
    return {
        "scene_count": len(scene_summaries),
        "methods": aggregate_methods,
    }


def _save_aggregate_plot(
    aggregate_summary: Dict[str, Any],
    results_root: Path,
) -> None:
    """Save the aggregate timing plot averaged across all scenes.

    Args:
        aggregate_summary: Aggregate timing-summary dictionary.
        results_root: Benchmark results root.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(aggregate_summary, dict), (
            "Expected `aggregate_summary` to be a dict. " f"{type(aggregate_summary)=}."
        )
        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )

    _validate_inputs()

    method_keys = _list_method_keys_from_aggregate(aggregate_summary=aggregate_summary)
    x_positions = np.arange(len(method_keys))
    visibility_means = np.asarray(
        [
            aggregate_summary["methods"][method_key]["visibility_mean_ms"]
            for method_key in method_keys
        ],
        dtype=np.float64,
    )
    other_means = np.asarray(
        [
            aggregate_summary["methods"][method_key]["other_mean_ms"]
            for method_key in method_keys
        ],
        dtype=np.float64,
    )
    total_means = np.asarray(
        [
            aggregate_summary["methods"][method_key]["total_mean_ms"]
            for method_key in method_keys
        ],
        dtype=np.float64,
    )
    total_stds = np.asarray(
        [
            aggregate_summary["methods"][method_key]["total_std_ms"]
            for method_key in method_keys
        ],
        dtype=np.float64,
    )
    method_labels = [
        (
            f"{aggregate_summary['methods'][method_key]['display_label']}\n"
            f"{aggregate_summary['methods'][method_key]['relative_to_v1_total_mean']:.2f}x"
        )
        for method_key in method_keys
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        x_positions,
        visibility_means,
        color="#1f77b4",
        label="Visibility",
    )
    axis.bar(
        x_positions,
        other_means,
        bottom=visibility_means,
        color="#ff7f0e",
        label="Other",
    )
    axis.errorbar(
        x_positions,
        total_means,
        yerr=total_stds,
        fmt="none",
        ecolor="#0f172a",
        capsize=5,
    )
    for method_idx, total_mean in enumerate(total_means):
        axis.text(
            x_positions[method_idx],
            total_mean + max(total_stds[method_idx], 1.0) + 2.0,
            f"{total_mean:.1f}±{total_stds[method_idx]:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(method_labels)
    axis.set_ylabel("Time (ms)")
    axis.set_title("Average Extraction Timing Across 30 GSO-CLOD Scenes")
    axis.grid(axis="y", color="#dbe4ee")
    axis.legend()
    figure.tight_layout()
    output_path = results_root / "average_timing_bar_plot.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _list_method_keys(
    scene_payload: Dict[str, Any],
) -> List[str]:
    """List the method keys for one scene payload in display order.

    Args:
        scene_payload: Cached scene payload dictionary.

    Returns:
        Ordered method-key list.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )

    _validate_inputs()

    ordered_keys = [
        "texel_visibility_v1",
        "texel_visibility_v2",
        "open3d_cpu",
        "open3d_gpu",
    ]
    return [
        method_key
        for method_key in ordered_keys
        if method_key in scene_payload["methods"]
    ]


def _list_method_keys_from_summary(
    scene_summary: Dict[str, Any],
) -> List[str]:
    """List the method keys for one scene summary in display order.

    Args:
        scene_summary: Scene-summary dictionary.

    Returns:
        Ordered method-key list.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_summary, dict), (
            "Expected `scene_summary` to be a dict. " f"{type(scene_summary)=}."
        )

    _validate_inputs()

    ordered_keys = [
        "texel_visibility_v1",
        "texel_visibility_v2",
        "open3d_cpu",
        "open3d_gpu",
    ]
    return [
        method_key
        for method_key in ordered_keys
        if method_key in scene_summary["methods"]
    ]


def _list_method_keys_from_aggregate(
    aggregate_summary: Dict[str, Any],
) -> List[str]:
    """List the aggregate method keys in display order.

    Args:
        aggregate_summary: Aggregate timing-summary dictionary.

    Returns:
        Ordered method-key list.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(aggregate_summary, dict), (
            "Expected `aggregate_summary` to be a dict. " f"{type(aggregate_summary)=}."
        )

    _validate_inputs()

    ordered_keys = [
        "texel_visibility_v1",
        "texel_visibility_v2",
        "open3d_cpu",
        "open3d_gpu",
    ]
    return [
        method_key
        for method_key in ordered_keys
        if method_key in aggregate_summary["methods"]
    ]


def _load_rgb_image(
    image_path: Path,
) -> np.ndarray:
    """Load one RGB image from disk.

    Args:
        image_path: Image filepath.

    Returns:
        RGB image array `[H, W, 3]` with dtype `uint8`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(image_path, Path), (
            "Expected `image_path` to be a `Path`. " f"{type(image_path)=}."
        )
        assert image_path.is_file(), (
            "Expected `image_path` to be a file. " f"{image_path=}"
        )

    _validate_inputs()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr is not None, (
        "Expected OpenCV to load the image successfully. " f"{image_path=}"
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _open3d_gpu_device() -> Optional[o3d.core.Device]:
    """Return the supported non-CPU Open3D device for this benchmark, if any.

    Args:
        None.

    Returns:
        Supported non-CPU Open3D device or `None`.
    """

    if hasattr(o3d.core, "sycl") and o3d.core.sycl.is_available():
        return o3d.core.Device("SYCL:0")
    return None


def _select_cuda_device() -> torch.device:
    """Select the CUDA device with the most free memory.

    Args:
        None.

    Returns:
        Selected CUDA device.
    """

    assert torch.cuda.is_available(), (
        "Expected CUDA to be available when selecting a CUDA device. "
        f"{torch.cuda.is_available()=}"
    )
    best_device_index = 0
    best_free_bytes = -1
    for device_index in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        if free_bytes > best_free_bytes:
            best_free_bytes = free_bytes
            best_device_index = device_index
    return torch.device(f"cuda:{best_device_index}")
