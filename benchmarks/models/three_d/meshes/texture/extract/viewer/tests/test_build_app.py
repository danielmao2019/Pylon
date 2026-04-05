"""Startup tests for the texture-extraction benchmark viewer."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from benchmarks.models.three_d.meshes.texture.extract.viewer.build_app import build_app
from benchmarks.models.three_d.meshes.texture.extract.viewer.layout.components import (
    RIGHT_PANEL_ID,
    SCENE_RADIO_ID,
)


def test_build_app_loads_cached_results_bundle(tmp_path: Path) -> None:
    """Build the Dash app from one tiny cached results bundle.

    Args:
        tmp_path: Temporary pytest path.

    Returns:
        None.
    """

    results_root = tmp_path / "results"
    scene_root = results_root / "demo_scene"
    scene_root.mkdir(parents=True)
    payload = {
        "scene_name": "demo_scene",
        "source_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "reference_texture_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "mesh_vertices": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "mesh_faces": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "mesh_vertex_uv": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "mesh_face_uvs": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "methods": {
            "texel_visibility_v1": _build_method_payload(
                display_label="Texel Visibility V1",
            ),
            "texel_visibility_v2": _build_method_payload(
                display_label="Texel Visibility V2",
            ),
            "open3d_cpu": _build_method_payload(
                display_label="Open3D CPU",
            ),
        },
    }
    torch.save(payload, scene_root / "scene_payload.pt")
    index_payload = {
        "scene_names": ["demo_scene"],
        "default_scene_name": "demo_scene",
        "open3d_gpu_supported": False,
        "aggregate_summary": {"scene_count": 1, "methods": {}},
    }
    (results_root / "benchmark_index.json").write_text(
        json.dumps(index_payload),
        encoding="utf-8",
    )

    app = build_app(results_root=results_root)

    assert app.layout is not None, f"{app.layout=}"
    component_ids = _collect_component_ids(component=app.layout)
    assert SCENE_RADIO_ID in component_ids, f"{component_ids=}"
    assert RIGHT_PANEL_ID in component_ids, f"{component_ids=}"


def _build_method_payload(
    display_label: str,
) -> Dict[str, Any]:
    """Build one tiny method payload for app-startup tests.

    Args:
        display_label: Method display label.

    Returns:
        Minimal cached method payload.
    """

    return {
        "display_label": display_label,
        "timings_ms": {
            "visibility": [1.0, 1.0, 1.0],
            "other": [2.0, 2.0, 2.0],
            "total": [3.0, 3.0, 3.0],
        },
        "timing_summary": {
            "visibility_mean_ms": 1.0,
            "visibility_std_ms": 0.0,
            "other_mean_ms": 2.0,
            "other_std_ms": 0.0,
            "total_mean_ms": 3.0,
            "total_std_ms": 0.0,
            "relative_to_v1_total_mean": 1.0,
        },
        "uv_texture_map": torch.zeros((4, 4, 3), dtype=torch.float32),
        "supported": True,
    }


def _collect_component_ids(component: object) -> List[Any]:
    """Collect Dash component ids recursively from one layout tree.

    Args:
        component: Dash component tree root.

    Returns:
        Flat component-id list.
    """

    component_ids = []
    component_id = getattr(component, "id", None)
    if component_id is not None:
        component_ids.append(component_id)
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            component_ids.extend(_collect_component_ids(component=child))
    elif children is not None:
        component_ids.extend(_collect_component_ids(component=children))
    return component_ids
