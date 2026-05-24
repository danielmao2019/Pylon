"""Point display response APIs."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import torch

from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.atomic_displays.points.ts.backend.core_points_display import (
    create_points_display_response,
)
from data.viewer.utils.atomic_displays.points.ts.backend.schemas.display_response import (
    ColorPCDisplayResponse,
    SegmentationPCDisplayResponse,
)
from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb


def create_color_pc_display_response(
    slot_id: str,
    title: str,
    point_cloud_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> ColorPCDisplayResponse:
    """Create a color point-cloud display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        point_cloud_path: Point-cloud artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Color point-cloud display response.
    """
    return create_points_display_response(
        response_type=ColorPCDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="color_pc",
        point_cloud_path=point_cloud_path,
        meta_info=meta_info,
    )


def create_segmentation_pc_display_response(
    slot_id: str,
    title: str,
    segmentation_pc_path: Optional[str],
    original_overlay_path: Optional[str],
) -> SegmentationPCDisplayResponse:
    """Create a segmentation point-cloud display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        segmentation_pc_path: Segmentation point-cloud artifact path.
        original_overlay_path: Original scene artifact path.

    Returns:
        Segmentation point-cloud display response.
    """
    if segmentation_pc_path is None:
        class_id_to_rgb = {}
        colorized_segmentation_pc_path = None
    else:
        segmentation_pc = load_point_cloud(filepath=segmentation_pc_path, device="cpu")
        class_ids = _segmentation_pc_class_ids(segmentation_pc)
        if class_ids is None:
            class_id_to_rgb = {}
            colorized_segmentation_pc_path = segmentation_pc_path
        else:
            class_id_to_rgb = map_class_ids_to_rgb(
                class_ids=torch.unique(class_ids),
            )
            colorized_segmentation_pc_path = _map_segmentation_pc_to_rgb(
                segmentation_pc_path=segmentation_pc_path,
                class_id_to_rgb=class_id_to_rgb,
            )
    response = create_points_display_response(
        response_type=SegmentationPCDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="segmentation_pc",
        point_cloud_path=colorized_segmentation_pc_path,
        meta_info=_build_segmentation_pc_meta_info(
            class_id_to_rgb=class_id_to_rgb,
        ),
    )
    if original_overlay_path is not None:
        response.original_overlay_url = "/api/artifacts?%s" % urlencode(
            {"path": original_overlay_path},
        )
    return response


def _map_segmentation_pc_to_rgb(
    segmentation_pc_path: str,
    class_id_to_rgb: Dict[int, Tuple[int, int, int]],
) -> str:
    """Write a backend-colorized segmentation point-cloud resource.

    Args:
        segmentation_pc_path: Class-labeled segmentation point-cloud path.
        class_id_to_rgb: Mapping from class id to RGB color tuple.

    Returns:
        Colorized point-cloud path.
    """
    assert isinstance(segmentation_pc_path, str), (
        "Segmentation point-cloud path must be a string. segmentation_pc_path=%r"
        % segmentation_pc_path
    )
    assert isinstance(class_id_to_rgb, dict), (
        "Class color mapping must be a dict. class_id_to_rgb=%r" % class_id_to_rgb
    )

    segmentation_pc = load_point_cloud(filepath=segmentation_pc_path, device="cpu")
    label = _segmentation_pc_class_ids(segmentation_pc).to(torch.int64)
    rgb = torch.zeros(
        (segmentation_pc.num_points, 3),
        dtype=torch.float32,
        device=segmentation_pc.xyz.device,
    )
    for class_id, color in class_id_to_rgb.items():
        rgb[label == int(class_id)] = torch.tensor(
            color,
            dtype=torch.float32,
            device=segmentation_pc.xyz.device,
        )
    colorized_pc = PointCloud(
        xyz=segmentation_pc.xyz,
        data={
            **{
                field_name: getattr(segmentation_pc, field_name)
                for field_name in segmentation_pc.field_names()
                if field_name not in {"xyz", "rgb", "colors"}
            },
            "rgb": rgb,
        },
    )
    output_path = _colorized_segmentation_pc_path(
        segmentation_pc_path=segmentation_pc_path,
    )
    save_point_cloud(pc=colorized_pc, output_filepath=str(output_path))
    return str(output_path)


def _colorized_segmentation_pc_path(segmentation_pc_path: str) -> Path:
    """Build the deterministic colorized point-cloud display path.

    Args:
        segmentation_pc_path: Class-labeled segmentation point-cloud path.

    Returns:
        Colorized point-cloud output path.
    """
    assert isinstance(segmentation_pc_path, str), (
        "Segmentation point-cloud path must be a string. segmentation_pc_path=%r"
        % segmentation_pc_path
    )

    path = Path(segmentation_pc_path)
    output_path = path.with_name("%s.viewer_colorized%s" % (path.stem, path.suffix))
    return output_path


def _segmentation_pc_class_ids(segmentation_pc: PointCloud) -> Optional[torch.Tensor]:
    """Return the class-id tensor from a loaded segmentation point cloud.

    Args:
        segmentation_pc: Loaded segmentation point cloud.

    Returns:
        Per-point class ids, or None when the point cloud is already colorized.
    """
    assert isinstance(segmentation_pc, PointCloud), (
        "Segmentation point cloud must be a PointCloud. segmentation_pc=%r"
        % segmentation_pc
    )
    field_names = segmentation_pc.field_names()
    if "label" in field_names:
        return segmentation_pc.label
    if "feat" in field_names:
        return segmentation_pc.feat
    return None


def _build_segmentation_pc_meta_info(
    class_id_to_rgb: Dict[int, Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Build factual class/color metadata from the class-to-RGB mapping.

    Args:
        class_id_to_rgb: Mapping from class id to RGB color tuple.

    Returns:
        Segmentation point-cloud metadata.
    """
    assert isinstance(class_id_to_rgb, dict), (
        "Class color mapping must be a dict. class_id_to_rgb=%r" % class_id_to_rgb
    )
    return {"class_id_to_rgb": class_id_to_rgb}
