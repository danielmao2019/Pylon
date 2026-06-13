"""Mesh display response APIs."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from data.structures.three_d.mesh.load import load_mesh
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.save import save_mesh
from data.viewer.utils.displays.mesh.ts.backend.core_mesh_display import (
    create_mesh_display_response_core,
)
from data.viewer.utils.displays.mesh.ts.backend.schemas.display_response import (
    ColorMeshDisplayResponse,
    HeatmapMeshDisplayResponse,
    SegmentationMeshDisplayResponse,
    SparseHeatmapMeshDisplayResponse,
)
from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
from data.viewer.utils.displays.utils.heatmap_colors import map_scalars_to_rgb


def create_color_mesh_display_response(
    input_path: Path,
    output_path: Path,
    url: str,
    slot_id: str,
    title: str,
    meta_info: Dict[str, Any],
) -> ColorMeshDisplayResponse:
    """Create a color mesh display response.

    Args:
        input_path: Input color mesh artifact path on disk.
        output_path: Output mesh artifact path on disk.
        url: Frontend resource URL pointing at the written output mesh.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        meta_info: Caller-provided renderer metadata.

    Returns:
        Color mesh display response.
    """
    base_response = create_mesh_display_response_core(
        input_path=input_path,
        output_path=output_path,
        url=url,
        slot_id=slot_id,
        title=title,
        meta_info=meta_info,
    )
    return ColorMeshDisplayResponse(
        slot_id=base_response.slot_id,
        title=base_response.title,
        url=base_response.url,
        meta_info=base_response.meta_info,
    )


def create_segmentation_mesh_display_response(
    input_path: Path,
    output_path: Path,
    url: str,
    slot_id: str,
    title: str,
    meta_info: Dict[str, Any],
) -> SegmentationMeshDisplayResponse:
    """Create a segmentation mesh display response.

    Args:
        input_path: Class-labeled segmentation mesh artifact path on disk.
        output_path: Output mesh artifact path on disk.
        url: Frontend resource URL pointing at the written output mesh.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        meta_info: Caller-provided renderer metadata.

    Returns:
        Segmentation mesh display response.
    """
    segmentation_mesh_class_ids = _read_segmentation_mesh_class_ids(
        input_path=input_path,
    )
    class_id_to_rgb = map_class_ids_to_rgb(
        class_ids=torch.unique(segmentation_mesh_class_ids),
    )
    _map_segmentation_mesh_to_rgb(
        input_path=input_path,
        output_path=output_path,
        class_id_to_rgb=class_id_to_rgb,
    )
    payload = dict(meta_info)
    payload.update(_build_segmentation_mesh_meta_info(class_id_to_rgb=class_id_to_rgb))
    base_response = create_mesh_display_response_core(
        input_path=output_path,
        output_path=output_path,
        url=url,
        slot_id=slot_id,
        title=title,
        meta_info=payload,
    )
    return SegmentationMeshDisplayResponse(
        slot_id=base_response.slot_id,
        title=base_response.title,
        url=base_response.url,
        meta_info=base_response.meta_info,
    )


def create_heatmap_mesh_display_response(
    input_path: Path,
    output_path: Path,
    url: str,
    slot_id: str,
    title: str,
    meta_info: Dict[str, Any],
) -> HeatmapMeshDisplayResponse:
    """Create a heatmap mesh display response.

    Args:
        input_path: Non-negative-scalar-labeled mesh artifact path on disk.
        output_path: Output mesh artifact path on disk.
        url: Frontend resource URL pointing at the written output mesh.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        meta_info: Caller-provided renderer metadata.

    Returns:
        Heatmap mesh display response.
    """
    heatmap_mesh_scalars = _read_heatmap_mesh_scalars(input_path=input_path)
    scalar_rgb = map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
    _map_heatmap_mesh_to_rgb(
        input_path=input_path,
        output_path=output_path,
        scalar_rgb=scalar_rgb,
    )
    payload = dict(meta_info)
    payload.update(_build_heatmap_mesh_meta_info(scalars=heatmap_mesh_scalars))
    base_response = create_mesh_display_response_core(
        input_path=output_path,
        output_path=output_path,
        url=url,
        slot_id=slot_id,
        title=title,
        meta_info=payload,
    )
    return HeatmapMeshDisplayResponse(
        slot_id=base_response.slot_id,
        title=base_response.title,
        url=base_response.url,
        meta_info=base_response.meta_info,
    )


def create_sparse_heatmap_mesh_display_response(
    input_path: Path,
    output_path: Path,
    url: str,
    slot_id: str,
    title: str,
    meta_info: Dict[str, Any],
) -> SparseHeatmapMeshDisplayResponse:
    """Create a sparse heatmap mesh display response.

    Emits ONLY the sparse heatmap wire resource to ``output_path`` -- a
    geometry reference plus the sparse (indices, values) delta; never
    materializes a full mesh. Intended for consumers that overlay many
    per-part heatmaps on one shared base mesh: the frontend renderer fetches
    the referenced geometry once and colors it by the per-part delta.

    Args:
        input_path: Sparse heatmap artifact path on disk. The file is a JSON
            object ``{"geometry_url": str, "indices": [int, ...],
            "values": [float, ...]}``; ``geometry_url`` is the resource URL
            of the shared base column mesh whose vertex domain the delta
            addresses, ``indices`` are vertex ids into that domain, and
            ``values`` are non-negative scalars.
        output_path: Output sparse heatmap artifact path on disk; the
            written file is the JSON object served to the frontend, with
            the same ``{"geometry_url", "indices", "values"}`` shape.
        url: Frontend resource URL pointing at the written output JSON.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        meta_info: Caller-provided renderer metadata.

    Returns:
        Sparse heatmap mesh display response.
    """
    indices, values = _read_sparse_heatmap_arrays(input_path=input_path)
    _write_sparse_heatmap_resource(
        input_path=input_path,
        output_path=output_path,
    )
    payload = dict(meta_info)
    payload.update(
        _build_sparse_heatmap_mesh_meta_info(indices=indices, values=values),
    )
    return SparseHeatmapMeshDisplayResponse(
        slot_id=slot_id,
        title=title,
        url=url,
        meta_info=payload,
    )


def _map_segmentation_mesh_to_rgb(
    input_path: Path,
    output_path: Path,
    class_id_to_rgb: Dict[int, Tuple[int, int, int]],
) -> None:
    """Read a segmentation mesh, colorize by class, and write to disk.

    Args:
        input_path: Class-labeled segmentation mesh artifact path on disk.
        output_path: Output mesh artifact path on disk.
        class_id_to_rgb: Mapping from class id to RGB color tuple.

    Returns:
        None.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)
    assert isinstance(
        output_path, Path
    ), "Expected `output_path` to be a `Path`. output_path=%r" % (output_path,)
    assert isinstance(
        class_id_to_rgb, dict
    ), "Expected `class_id_to_rgb` to be a `dict`. class_id_to_rgb=%r" % (
        class_id_to_rgb,
    )

    mesh = Mesh(**load_mesh(path=input_path))
    if mesh.texture_mode == "vertex_color":
        class_ids = _segmentation_mesh_per_vertex_class_ids(mesh=mesh)
        vertex_color = torch.zeros(
            (class_ids.shape[0], 3),
            dtype=torch.uint8,
        )
        for class_id, color in class_id_to_rgb.items():
            vertex_color[class_ids == int(class_id)] = torch.tensor(
                color,
                dtype=torch.uint8,
            )
        colorized_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_color=vertex_color,
        )
    elif mesh.texture_mode == "uv_texture_map":
        class_ids = _segmentation_mesh_per_texel_class_ids(mesh=mesh)
        height, width = class_ids.shape
        uv_texture_map = torch.zeros((height, width, 3), dtype=torch.uint8)
        for class_id, color in class_id_to_rgb.items():
            uv_texture_map[class_ids == int(class_id)] = torch.tensor(
                color,
                dtype=torch.uint8,
            )
        colorized_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            uv_texture_map=uv_texture_map,
            vertex_uv=mesh.vertex_uv,
            face_uvs=mesh.face_uvs,
            convention=mesh.convention,
        )
    else:
        raise ValueError(
            "Unsupported segmentation mesh texture representation. texture_mode=%r"
            % (mesh.texture_mode,),
        )
    save_mesh(mesh=colorized_mesh, output_path=output_path)


def _map_heatmap_mesh_to_rgb(
    input_path: Path,
    output_path: Path,
    scalar_rgb: torch.Tensor,
) -> None:
    """Read a heatmap mesh, replace its scalar storage with RGB, and write to disk.

    Args:
        input_path: Non-negative-scalar-labeled mesh artifact path on disk.
        output_path: Output mesh artifact path on disk.
        scalar_rgb: Colorized scalar tensor of shape `scalars.shape + (3,)`,
            uint8 dtype in `[0, 255]`.

    Returns:
        None.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)
    assert isinstance(
        output_path, Path
    ), "Expected `output_path` to be a `Path`. output_path=%r" % (output_path,)
    assert isinstance(
        scalar_rgb, torch.Tensor
    ), "Expected `scalar_rgb` to be a `torch.Tensor`. scalar_rgb=%r" % (scalar_rgb,)
    assert (
        scalar_rgb.dtype == torch.uint8
    ), "Expected `scalar_rgb` to be uint8. scalar_rgb.dtype=%r" % (scalar_rgb.dtype,)

    mesh = Mesh(**load_mesh(path=input_path))
    if mesh.texture_mode == "vertex_color":
        assert (
            scalar_rgb.ndim == 2 and scalar_rgb.shape[-1] == 3
        ), "Expected per-vertex `scalar_rgb` to be `[V, 3]`. scalar_rgb.shape=%r" % (
            scalar_rgb.shape,
        )
        colorized_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_color=scalar_rgb,
        )
    elif mesh.texture_mode == "uv_texture_map":
        assert (
            scalar_rgb.ndim == 3 and scalar_rgb.shape[-1] == 3
        ), "Expected per-texel `scalar_rgb` to be `[H, W, 3]`. scalar_rgb.shape=%r" % (
            scalar_rgb.shape,
        )
        colorized_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            uv_texture_map=scalar_rgb,
            vertex_uv=mesh.vertex_uv,
            face_uvs=mesh.face_uvs,
            convention=mesh.convention,
        )
    else:
        raise ValueError(
            "Unsupported heatmap mesh texture representation. texture_mode=%r"
            % (mesh.texture_mode,),
        )
    save_mesh(mesh=colorized_mesh, output_path=output_path)


def _build_segmentation_mesh_meta_info(
    class_id_to_rgb: Dict[int, Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Build factual class/color metadata from the class-to-RGB mapping.

    Args:
        class_id_to_rgb: Mapping from class id to RGB color tuple.

    Returns:
        Segmentation mesh metadata.
    """
    assert isinstance(
        class_id_to_rgb, dict
    ), "Expected `class_id_to_rgb` to be a `dict`. class_id_to_rgb=%r" % (
        class_id_to_rgb,
    )
    return {"class_id_to_rgb": class_id_to_rgb}


def _build_heatmap_mesh_meta_info(scalars: torch.Tensor) -> Dict[str, Any]:
    """Build scalar-range metadata from the input scalars.

    Args:
        scalars: Non-negative scalar tensor; per-vertex 1-D or per-texel 2-D.

    Returns:
        Heatmap mesh metadata containing scalar min/max.
    """
    assert isinstance(
        scalars, torch.Tensor
    ), "Expected `scalars` to be a `torch.Tensor`. scalars=%r" % (scalars,)
    assert (
        scalars.numel() > 0
    ), "Expected `scalars` to be non-empty. scalars.shape=%r" % (scalars.shape,)
    return {
        "scalar_min": float(scalars.min().item()),
        "scalar_max": float(scalars.max().item()),
    }


def _read_segmentation_mesh_class_ids(input_path: Path) -> torch.Tensor:
    """Read per-vertex or per-texel class ids from a segmentation mesh.

    Args:
        input_path: Class-labeled segmentation mesh artifact path on disk.

    Returns:
        Integer class-id tensor; per-vertex 1-D `[V]` or per-texel 2-D `[H, W]`.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)

    mesh = Mesh(**load_mesh(path=input_path))
    if mesh.texture_mode == "vertex_color":
        return _segmentation_mesh_per_vertex_class_ids(mesh=mesh)
    if mesh.texture_mode == "uv_texture_map":
        return _segmentation_mesh_per_texel_class_ids(mesh=mesh)
    raise ValueError(
        "Unsupported segmentation mesh texture representation. texture_mode=%r"
        % (mesh.texture_mode,),
    )


def _read_heatmap_mesh_scalars(input_path: Path) -> torch.Tensor:
    """Read per-vertex or per-texel non-negative scalars from a heatmap mesh.

    Args:
        input_path: Non-negative-scalar-labeled mesh artifact path on disk.

    Returns:
        Non-negative scalar tensor; per-vertex 1-D `[V]` or per-texel 2-D `[H, W]`.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)

    mesh = Mesh(**load_mesh(path=input_path))
    if mesh.texture_mode == "vertex_color":
        assert (
            mesh.vertex_color is not None
        ), "Expected heatmap mesh vertex storage to be present. mesh.vertex_color=None"
        return mesh.vertex_color[:, 0]
    if mesh.texture_mode == "uv_texture_map":
        assert mesh.uv_texture_map is not None, (
            "Expected heatmap mesh texel storage to be present. "
            "mesh.uv_texture_map=None"
        )
        return mesh.uv_texture_map[..., 0]
    raise ValueError(
        "Unsupported heatmap mesh texture representation. texture_mode=%r"
        % (mesh.texture_mode,),
    )


def _segmentation_mesh_per_vertex_class_ids(mesh: Mesh) -> torch.Tensor:
    """Extract per-vertex class ids from a vertex-colored segmentation mesh.

    Args:
        mesh: Loaded segmentation mesh with per-vertex class-id storage.

    Returns:
        Integer per-vertex class-id tensor of shape `[V]`.
    """
    assert isinstance(mesh, Mesh), "Expected `mesh` to be a `Mesh`. mesh=%r" % (mesh,)
    assert (
        mesh.vertex_color is not None
    ), "Expected segmentation mesh vertex storage to be present. mesh.vertex_color=None"
    return mesh.vertex_color[:, 0].to(dtype=torch.int64)


def _segmentation_mesh_per_texel_class_ids(mesh: Mesh) -> torch.Tensor:
    """Extract per-texel class ids from a UV-textured segmentation mesh.

    Args:
        mesh: Loaded segmentation mesh with per-texel class-id storage.

    Returns:
        Integer per-texel class-id tensor of shape `[H, W]`.
    """
    assert isinstance(mesh, Mesh), "Expected `mesh` to be a `Mesh`. mesh=%r" % (mesh,)
    assert mesh.uv_texture_map is not None, (
        "Expected segmentation mesh texel storage to be present. "
        "mesh.uv_texture_map=None"
    )
    return mesh.uv_texture_map[..., 0].to(dtype=torch.int64)


def _read_sparse_heatmap_geometry_url(input_path: Path) -> str:
    """Read the shared-geometry reference from a sparse heatmap JSON file.

    Args:
        input_path: Sparse heatmap artifact path on disk. The file is a JSON
            object ``{"geometry_url": str, "indices": [int, ...],
            "values": [float, ...]}``.

    Returns:
        The ``geometry_url`` string: the resource URL of the shared base
        column mesh whose vertex domain the sparse delta addresses.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)

    with input_path.open("r") as fh:
        payload = json.load(fh)
    assert "geometry_url" in payload, (
        "Expected sparse heatmap JSON to carry a `geometry_url` geometry "
        "reference. input_path=%r payload_keys=%r"
        % (input_path, sorted(payload.keys()))
    )
    geometry_url = payload["geometry_url"]
    assert (
        isinstance(geometry_url, str) and len(geometry_url) > 0
    ), "Expected `geometry_url` to be a non-empty string. geometry_url=%r" % (
        geometry_url,
    )
    return geometry_url


def _read_sparse_heatmap_arrays(input_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read sparse heatmap (indices, values) arrays from a JSON file.

    Args:
        input_path: Sparse heatmap artifact path on disk. The file is a JSON
            object ``{"geometry_url": str, "indices": [int, ...],
            "values": [float, ...]}``.

    Returns:
        2-tuple ``(indices, values)`` of 1-D tensors with matching length.
        ``indices`` is int64 and non-negative; ``values`` is float32 and
        non-negative.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)

    with input_path.open("r") as fh:
        payload = json.load(fh)
    indices = torch.tensor(payload["indices"], dtype=torch.int64)
    values = torch.tensor(payload["values"], dtype=torch.float32)
    assert indices.ndim == 1, "Expected `indices` to be 1-D. indices.shape=%r" % (
        indices.shape,
    )
    assert values.ndim == 1, "Expected `values` to be 1-D. values.shape=%r" % (
        values.shape,
    )
    assert indices.shape[0] == values.shape[0], (
        "Expected `indices` and `values` to have matching length. "
        "indices.shape=%r values.shape=%r" % (indices.shape, values.shape)
    )
    assert bool((indices >= 0).all()), (
        "Expected `indices` to be non-negative. indices.min()=%r" % indices.min().item()
    )
    assert bool((values >= 0).all()), (
        "Expected `values` to be non-negative. values.min()=%r" % values.min().item()
    )
    return indices, values


def _write_sparse_heatmap_resource(input_path: Path, output_path: Path) -> None:
    """Read the geometry reference and (indices, values) delta from
    ``input_path`` and write the served wire resource to ``output_path``.

    Args:
        input_path: Sparse heatmap artifact path on disk. The file is a JSON
            object ``{"geometry_url": str, "indices": [int, ...],
            "values": [float, ...]}``.
        output_path: Output sparse heatmap artifact path on disk; the
            written file is the JSON object served to the frontend, with
            the same ``{"geometry_url", "indices", "values"}`` shape.

    Returns:
        None.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)
    assert isinstance(
        output_path, Path
    ), "Expected `output_path` to be a `Path`. output_path=%r" % (output_path,)

    geometry_url = _read_sparse_heatmap_geometry_url(input_path=input_path)
    indices, values = _read_sparse_heatmap_arrays(input_path=input_path)
    payload = {
        "geometry_url": geometry_url,
        "indices": indices.tolist(),
        "values": values.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(payload, fh)


def _build_sparse_heatmap_mesh_meta_info(
    indices: torch.Tensor,
    values: torch.Tensor,
) -> Dict[str, Any]:
    """Build scalar-range metadata from the sparse heatmap values.

    Args:
        indices: 1-D int64 tensor of vertex ids in the base mesh.
        values: 1-D float32 tensor of non-negative scalar values; same
            length as ``indices``.

    Returns:
        Sparse heatmap metadata containing scalar min/max. Range is empty
        when ``values`` is empty (no covered vertices); in that case both
        bounds default to ``0.0``.
    """
    assert isinstance(
        indices, torch.Tensor
    ), "Expected `indices` to be a `torch.Tensor`. indices=%r" % (indices,)
    assert isinstance(
        values, torch.Tensor
    ), "Expected `values` to be a `torch.Tensor`. values=%r" % (values,)
    if values.numel() == 0:
        return {"scalar_min": 0.0, "scalar_max": 0.0}
    return {
        "scalar_min": float(values.min().item()),
        "scalar_max": float(values.max().item()),
    }
