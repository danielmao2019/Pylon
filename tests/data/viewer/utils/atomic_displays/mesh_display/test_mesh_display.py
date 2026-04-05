"""Tests for mesh display functionality."""

from pathlib import Path

import plotly.graph_objects as go
import pytest
import torch
from dash import html

import data.viewer.utils.atomic_displays.mesh_display as mesh_display_module
from data.structures.three_d.mesh import Mesh
from data.viewer.utils.atomic_displays import create_mesh_display


def _build_uv_test_mesh() -> Mesh:
    """Build one small UV-textured mesh for shared display tests.

    Args:
        None.

    Returns:
        UV-textured mesh.
    """

    return Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        uv_texture_map=torch.tensor(
            [
                [[255, 0, 0], [255, 0, 0]],
                [[0, 0, 255], [0, 0, 255]],
            ],
            dtype=torch.uint8,
        ),
        vertex_uv=torch.tensor(
            [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="top_left",
    )


def test_create_mesh_display_vertex_color_mesh() -> None:
    """Create one mesh figure from per-vertex colors.

    Args:
        None.

    Returns:
        None.
    """

    mesh = Mesh(
        vertices=torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        vertex_color=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )

    figure = create_mesh_display(
        mesh=mesh,
        title="Vertex Mesh",
    )

    assert isinstance(figure, go.Figure), f"{type(figure)=}"
    assert figure.layout.title.text == "Vertex Mesh", f"{figure.layout.title.text=}"
    assert len(figure.data) == 1, f"{len(figure.data)=}"
    assert figure.data[0].type == "mesh3d", f"{figure.data[0].type=}"
    assert figure.data[0].vertexcolor == (
        "rgb(255,0,0)",
        "rgb(0,255,0)",
        "rgb(0,0,255)",
    ), f"{figure.data[0].vertexcolor=}"
    assert (
        pytest.approx(float(min(figure.data[0].x)), abs=1e-6) == 1.0
    ), f"{figure.data[0].x=}"
    assert (
        pytest.approx(float(max(figure.data[0].x)), abs=1e-6) == 3.0
    ), f"{figure.data[0].x=}"
    assert tuple(float(value) for value in figure.layout.scene.xaxis.range) == (
        1.0,
        3.0,
    ), f"{figure.layout.scene.xaxis.range=}"
    assert tuple(float(value) for value in figure.layout.scene.yaxis.range) == (
        0.0,
        2.0,
    ), f"{figure.layout.scene.yaxis.range=}"
    assert tuple(float(value) for value in figure.layout.scene.zaxis.range) == (
        0.0,
        0.0,
    ), f"{figure.layout.scene.zaxis.range=}"
    assert figure.layout.scene.aspectmode == "data", (
        "Expected mesh displays to preserve raw axis proportions instead of "
        "forcing cube-normalized display scaling. "
        f"{figure.layout.scene.aspectmode=}"
    )
    assert figure.layout.uirevision == "mesh-display-camera", (
        "Expected vertex-color mesh displays to use a stable Plotly "
        "`uirevision` so camera state persists across figure updates. "
        f"{figure.layout.uirevision=}"
    )


def test_build_mesh_view_bounds_computes_camera_coordinate_scale() -> None:
    """Build one camera-coordinate scale from the positive mesh axis spans.

    Args:
        None.

    Returns:
        None.
    """

    volumetric_vertices = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [5.0, 2.0, 3.0],
            [1.0, 8.0, 3.0],
            [1.0, 2.0, 15.0],
        ],
        dtype=torch.float32,
    )
    volumetric_view_bounds = mesh_display_module.build_mesh_view_bounds(
        vertices=volumetric_vertices,
    )

    assert pytest.approx(
        volumetric_view_bounds["camera_coordinate_scale"], abs=1.0e-6
    ) == (4.0 * 6.0 * 12.0) ** (1.0 / 3.0), f"{volumetric_view_bounds=}"

    planar_vertices = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [5.0, 2.0, 3.0],
            [1.0, 8.0, 3.0],
        ],
        dtype=torch.float32,
    )
    planar_view_bounds = mesh_display_module.build_mesh_view_bounds(
        vertices=planar_vertices,
    )

    assert (
        pytest.approx(planar_view_bounds["camera_coordinate_scale"], abs=1.0e-6)
        == (4.0 * 6.0) ** 0.5
    ), f"{planar_view_bounds=}"


def test_create_mesh_display_does_not_modify_input_mesh_data() -> None:
    """Creating displays should leave the caller's mesh tensors unchanged.

    Args:
        None.

    Returns:
        None.
    """

    vertex_color_mesh = Mesh(
        vertices=torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        vertex_color=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )
    vertex_color_vertices_before = vertex_color_mesh.vertices.clone()
    vertex_color_faces_before = vertex_color_mesh.faces.clone()
    vertex_color_values_before = vertex_color_mesh.vertex_color.clone()

    create_mesh_display(
        mesh=vertex_color_mesh,
        title="Vertex Mesh",
    )

    assert torch.equal(vertex_color_mesh.vertices, vertex_color_vertices_before), (
        "Expected vertex-color mesh display creation to preserve input vertices. "
        f"{vertex_color_mesh.vertices=} {vertex_color_vertices_before=}"
    )
    assert torch.equal(vertex_color_mesh.faces, vertex_color_faces_before), (
        "Expected vertex-color mesh display creation to preserve input faces. "
        f"{vertex_color_mesh.faces=} {vertex_color_faces_before=}"
    )
    assert torch.equal(vertex_color_mesh.vertex_color, vertex_color_values_before), (
        "Expected vertex-color mesh display creation to preserve input colors. "
        f"{vertex_color_mesh.vertex_color=} {vertex_color_values_before=}"
    )

    uv_mesh = _build_uv_test_mesh()
    uv_vertices_before = uv_mesh.vertices.clone()
    uv_faces_before = uv_mesh.faces.clone()
    uv_texture_before = uv_mesh.uv_texture_map.clone()
    uv_vertex_uv_before = uv_mesh.vertex_uv.clone()
    uv_face_uvs_before = uv_mesh.face_uvs.clone()

    create_mesh_display(
        mesh=uv_mesh,
        title="UV Mesh",
        component_id="uv-mesh-viewer",
    )

    assert torch.equal(uv_mesh.vertices, uv_vertices_before), (
        "Expected UV mesh display creation to preserve input vertices. "
        f"{uv_mesh.vertices=} {uv_vertices_before=}"
    )
    assert torch.equal(uv_mesh.faces, uv_faces_before), (
        "Expected UV mesh display creation to preserve input faces. "
        f"{uv_mesh.faces=} {uv_faces_before=}"
    )
    assert torch.equal(uv_mesh.uv_texture_map, uv_texture_before), (
        "Expected UV mesh display creation to preserve input texture values. "
        f"{uv_mesh.uv_texture_map=} {uv_texture_before=}"
    )
    assert torch.equal(uv_mesh.vertex_uv, uv_vertex_uv_before), (
        "Expected UV mesh display creation to preserve input UV coordinates. "
        f"{uv_mesh.vertex_uv=} {uv_vertex_uv_before=}"
    )
    assert torch.equal(uv_mesh.face_uvs, uv_face_uvs_before), (
        "Expected UV mesh display creation to preserve input face UV indices. "
        f"{uv_mesh.face_uvs=} {uv_face_uvs_before=}"
    )


def test_create_mesh_display_uv_texture_mesh() -> None:
    """Create one mesh iframe from a UV texture map.

    Args:
        None.

    Returns:
        None.
    """

    display = create_mesh_display(
        mesh=_build_uv_test_mesh(),
        title="UV Mesh",
        component_id="uv-mesh-viewer",
    )

    assert isinstance(display, html.Iframe), f"{type(display)=}"
    assert display.id == "uv-mesh-viewer", f"{display.id=}"
    assert isinstance(display.srcDoc, str), f"{type(display.srcDoc)=}"
    assert "three.min.js" in display.srcDoc, f"{display.srcDoc[:160]=}"
    assert "const viewerConfig =" in display.srcDoc, f"{display.srcDoc[:240]=}"
    assert "MeshBasicMaterial" in display.srcDoc, f"{display.srcDoc[:240]=}"
    assert "controls.enablePan = true;" in display.srcDoc, f"{display.srcDoc[:240]=}"
    assert "texture.flipY = true;" in display.srcDoc, f"{display.srcDoc[:240]=}"
    assert "[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]" in display.srcDoc, (
        "Expected the viewer payload to contain OBJ-style UV values after "
        "conversion from the mesh convention. "
        f"{display.srcDoc[:400]=}"
    )
    assert (
        "window.__threejsCameraSyncViewBounds = viewerConfig.meshViewBounds;"
        in display.srcDoc
    ), f"{display.srcDoc[:320]=}"
    assert (
        "const cameraPersistenceStorageKey =" in display.srcDoc
    ), f"{display.srcDoc[:480]=}"
    assert (
        '"mesh-display-camera:" + "uv-mesh-viewer"' in display.srcDoc
    ), f"{display.srcDoc[:640]=}"
    assert "window.localStorage.getItem(" in display.srcDoc, f"{display.srcDoc[:640]=}"
    assert "window.localStorage.setItem(" in display.srcDoc, f"{display.srcDoc[:640]=}"
    assert "data:image/png;base64," in display.srcDoc, f"{display.srcDoc[:240]=}"


def test_create_mesh_display_uv_texture_mesh_uses_sibling_javascript_templates() -> (
    None
):
    """Use sibling JavaScript template files for the UV-texture viewer.

    Args:
        None.

    Returns:
        None.
    """

    module_dir = Path(mesh_display_module.__file__).resolve().parent
    viewer_script_path = module_dir / "mesh_display_textured_viewer.js"

    display = create_mesh_display(
        mesh=_build_uv_test_mesh(),
        title="UV Mesh",
        component_id="uv-mesh-viewer",
        camera_sync_group="sync-group",
    )

    assert viewer_script_path.is_file(), f"{viewer_script_path=}"
    assert isinstance(display, html.Iframe), f"{type(display)=}"
    assert "window.__threejsCameraSync" in display.srcDoc, f"{display.srcDoc[:320]=}"
    assert "cameraSyncGroup: 'sync-group'" in display.srcDoc, f"{display.srcDoc[:400]=}"
    assert display.__dict__["data-camera-sync-group"] == "sync-group", f"{display=}"
    assert (
        display.__dict__["data-camera-sync-viewer-id"] == "uv-mesh-viewer"
    ), f"{display=}"


def test_create_mesh_display_uv_texture_mesh_normalizes_defaults() -> None:
    """Normalize UV-texture titles, component ids, and sync groups.

    Args:
        None.

    Returns:
        None.
    """

    display = create_mesh_display(
        mesh=_build_uv_test_mesh(),
        title="  UV Mesh  ",
        camera_sync_group="  sync-group  ",
    )

    assert isinstance(display, html.Iframe), f"{type(display)=}"
    assert display.id == "mesh-display-uv-mesh", f"{display.id=}"
    assert "<title>UV Mesh</title>" in display.srcDoc, f"{display.srcDoc[:200]=}"
    assert (
        "viewerId: 'mesh-display-uv-mesh'" in display.srcDoc
    ), f"{display.srcDoc[:400]=}"


def test_build_threejs_viewer_html_includes_extra_script_urls() -> None:
    """Build a shared Three.js iframe shell with optional loader scripts.

    Args:
        None.

    Returns:
        None.
    """

    html_text = mesh_display_module.build_threejs_viewer_html(
        title="Shared Viewer",
        viewer_script="console.log('viewer');",
        extra_script_urls=["https://example.com/OBJLoader.js"],
    )

    assert "<title>Shared Viewer</title>" in html_text, f"{html_text[:240]=}"
    assert "three.min.js" in html_text, f"{html_text[:240]=}"
    assert "OrbitControls.js" in html_text, f"{html_text[:240]=}"
    assert "https://example.com/OBJLoader.js" in html_text, f"{html_text[:240]=}"
    assert "console.log('viewer');" in html_text, f"{html_text[:240]=}"


def test_create_mesh_display_rejects_invalid_mesh_type() -> None:
    """Reject non-mesh inputs.

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError):
        create_mesh_display(
            mesh="not-a-mesh",
            title="Invalid",
        )
