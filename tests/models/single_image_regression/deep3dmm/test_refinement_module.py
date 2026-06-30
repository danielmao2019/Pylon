"""Tests for Deep3DMM model export behavior."""

from pathlib import Path

import pytest
import torch
from project.models.single_image_regression.deep3dmm.model import (
    Deep3dmmModel,
)

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)


def test_write_obj_with_uv_texture_map_keeps_texture_image_rows_before_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Check the Deep3DMM OBJ export keeps texture-image rows unchanged.

    Args:
        monkeypatch: Pytest monkeypatch helper.
        tmp_path: Temporary filesystem path for the export target.

    Returns:
        None.
    """

    captured = {}

    def fake_save(self: Mesh, path: Path) -> None:
        """Capture the mesh that Deep3DMM export passes to save.

        Args:
            path: OBJ output path passed by the export helper.

        Returns:
            None.
        """

        captured["mesh"] = self
        captured["path"] = path

    monkeypatch.setattr(Mesh, "save", fake_save)

    output_obj_path = tmp_path / "mesh.obj"
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    verts_uvs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    faces_uvs = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    uv_texture_map = torch.tensor(
        [
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
            [[0.70, 0.80, 0.90], [1.00, 0.00, 0.10]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    Deep3dmmModel._write_obj_with_uv_texture_map(
        output_obj_path=output_obj_path,
        vertices=vertices,
        faces=faces,
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        uv_texture_map=uv_texture_map,
    )

    assert captured["path"] == output_obj_path, f"{captured=}"
    assert isinstance(
        captured["mesh"].texture, MeshTextureUVTextureMap
    ), f"{type(captured['mesh'].texture)=}"
    assert (
        captured["mesh"].texture.convention == "obj"
    ), f"{captured['mesh'].texture.convention=}"
    assert captured["mesh"].verts.requires_grad, (
        "Expected Deep3DMM export to leave vertex tensor ownership to Mesh.save. "
        f"{captured['mesh'].verts.requires_grad=}"
    )
    assert captured["mesh"].texture.verts_uvs.requires_grad, (
        "Expected Deep3DMM export to leave vertex-uv tensor ownership to Mesh.save. "
        f"{captured['mesh'].texture.verts_uvs.requires_grad=}"
    )
    assert captured["mesh"].texture.uv_texture_map.requires_grad, (
        "Expected Deep3DMM export to leave texture tensor ownership to Mesh.save. "
        f"{captured['mesh'].texture.uv_texture_map.requires_grad=}"
    )
    assert torch.allclose(
        captured["mesh"].texture.uv_texture_map,
        uv_texture_map,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{captured['mesh'].texture.uv_texture_map=} {uv_texture_map=}"
