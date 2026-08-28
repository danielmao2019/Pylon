"""Unit tests for the seam-shift-at-load + collapse-on-save OBJ round trip."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.canonicalize import (
    collapse_seam_shifted_uv_rows,
    shift_seam_crossing_faces_to_seam_safe,
)


def _write_seamed_uv_obj(directory: Path) -> Path:
    """Write one hand-written seamed UV OBJ with a sibling MTL and texture PNG.

    The OBJ carries one seam-crossing face (corner u's {0.97, 0.02, 0.05}
    straddle the cylindrical wrap) and one non-seam face; the two faces share
    two vt rows, so loading forks those rows into seam-shifted siblings and
    saving must collapse them back.

    Args:
        directory: Directory in which to write the OBJ / MTL / PNG.

    Returns:
        Path to the written OBJ file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    obj_path = directory / "seam.obj"
    mtl_path = directory / "seam.mtl"
    texture_path = directory / "seam_texture.png"

    obj_path.write_text(
        "mtllib seam.mtl\n"
        "usemtl material0\n"
        "v 0.000000 0.000000 0.000000\n"
        "v 1.000000 0.000000 0.000000\n"
        "v 1.000000 1.000000 0.000000\n"
        "v 0.000000 1.000000 0.000000\n"
        "vt 0.970000 0.200000\n"
        "vt 0.020000 0.250000\n"
        "vt 0.050000 0.800000\n"
        "vt 0.100000 0.300000\n"
        "f 1/1 2/2 3/3\n"
        "f 2/2 3/3 4/4\n",
        encoding="utf-8",
    )
    mtl_path.write_text(
        "newmtl material0\nmap_Kd seam_texture.png\n",
        encoding="utf-8",
    )
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(str(texture_path))
    return obj_path


def _extract_lines_with_prefix(obj_path: Path, prefix: str) -> str:
    """Extract the lines of one OBJ file that start with a given prefix.

    Args:
        obj_path: OBJ filepath to read.
        prefix: Line prefix to select, e.g. ``"vt "`` or ``"f "``.

    Returns:
        The selected lines joined with newlines, in file order.
    """
    lines = obj_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if line.startswith(prefix))


def test_load_save_obj_with_seam_face_is_byte_identical(tmp_path: Path) -> None:
    """Load a hand-written seamed UV OBJ, save it back, and assert byte equality of the resulting vt / f lines — exercises the seam-shift-at-load + collapse-on-save round-trip.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    source_obj_path = _write_seamed_uv_obj(directory=tmp_path / "source")
    mesh = Mesh.load(path=source_obj_path)

    saved_obj_path = tmp_path / "saved" / "seam.obj"
    mesh.save(path=saved_obj_path)

    source_vt_lines = _extract_lines_with_prefix(obj_path=source_obj_path, prefix="vt ")
    saved_vt_lines = _extract_lines_with_prefix(obj_path=saved_obj_path, prefix="vt ")
    assert saved_vt_lines == source_vt_lines, (
        "Expected the saved vt lines to be byte-identical to the source OBJ's. "
        f"{saved_vt_lines=} {source_vt_lines=}"
    )

    source_f_lines = _extract_lines_with_prefix(obj_path=source_obj_path, prefix="f ")
    saved_f_lines = _extract_lines_with_prefix(obj_path=saved_obj_path, prefix="f ")
    assert saved_f_lines == source_f_lines, (
        "Expected the saved f lines to be byte-identical to the source OBJ's. "
        f"{saved_f_lines=} {source_f_lines=}"
    )


def test_load_promotes_seam_crossing_face_to_seam_safe_canonical(
    tmp_path: Path,
) -> None:
    """After load, every face of a seamed mesh is non-wrapping (its largest cyclic gap over verts_uvs[faces_uvs[f]] is the wraparound gap), the seam-safe canonical form.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    obj_path = _write_seamed_uv_obj(directory=tmp_path)
    mesh = Mesh.load(path=obj_path)

    face_corner_u = mesh.texture.verts_uvs[mesh.texture.faces_uvs, 0]
    for face_index in range(int(face_corner_u.shape[0])):
        sorted_u = face_corner_u[face_index].sort().values
        interior_gaps = sorted_u[1:] - sorted_u[:-1]
        wraparound_gap = sorted_u[0] + 1.0 - sorted_u[-1]
        assert bool((interior_gaps <= wraparound_gap).all().item()), (
            "Expected every loaded face to be non-wrapping: its largest cyclic "
            "gap must be the wraparound gap. "
            f"{face_index=} {sorted_u=} {interior_gaps=} {wraparound_gap=}"
        )


def test_a_vt_row_shared_between_a_shifted_and_an_unshifted_face_is_forked() -> None:
    """Fork a vt row referenced by both a seam-crossing face and a stable face.

    Args:
        None.

    Returns:
        None.
    """
    source_verts_uvs = torch.tensor(
        [
            [0.97, 0.20],
            [0.02, 0.25],
            [0.05, 0.80],
            [0.30, 0.30],
            [0.40, 0.40],
        ],
        dtype=torch.float32,
    )
    source_faces_uvs = torch.tensor([[0, 1, 2], [1, 3, 4]], dtype=torch.int64)

    canonical_verts_uvs, canonical_faces_uvs = shift_seam_crossing_faces_to_seam_safe(
        verts_uvs=source_verts_uvs,
        faces_uvs=source_faces_uvs,
    )

    appended_row_index = int(source_verts_uvs.shape[0])
    expected_appended_row = source_verts_uvs[1].clone()
    expected_appended_row[0] = expected_appended_row[0] + 1.0
    assert canonical_verts_uvs.shape == (6, 2), (
        "Expected exactly one shared vt row to fork. " f"{canonical_verts_uvs.shape=}"
    )
    assert torch.allclose(
        canonical_verts_uvs[appended_row_index],
        expected_appended_row,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{canonical_verts_uvs[appended_row_index]=} {expected_appended_row=}"
    assert int(canonical_faces_uvs[0, 1].item()) == appended_row_index, (
        "Expected the seam-crossing face's shifting corner to reference the "
        "appended row. "
        f"{canonical_faces_uvs=}"
    )
    assert int(canonical_faces_uvs[1, 0].item()) == 1, (
        "Expected the stable face's shared corner to keep referencing the "
        "source row. "
        f"{canonical_faces_uvs=}"
    )


def test_a_mesh_with_no_seam_crossing_face_comes_back_unchanged() -> None:
    """Return an already seam-safe UV table and its face indices unchanged.

    Args:
        None.

    Returns:
        None.
    """
    source_verts_uvs = torch.tensor(
        [
            [0.10, 0.10],
            [0.35, 0.20],
            [0.45, 0.70],
            [0.20, 0.80],
        ],
        dtype=torch.float32,
    )
    source_faces_uvs = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)

    canonical_verts_uvs, canonical_faces_uvs = shift_seam_crossing_faces_to_seam_safe(
        verts_uvs=source_verts_uvs,
        faces_uvs=source_faces_uvs,
    )

    assert torch.equal(
        canonical_verts_uvs,
        source_verts_uvs,
    ), f"{canonical_verts_uvs=} {source_verts_uvs=}"
    assert torch.equal(
        canonical_faces_uvs,
        source_faces_uvs,
    ), f"{canonical_faces_uvs=} {source_faces_uvs=}"


def test_save_collapses_seam_shifted_uv_rows() -> None:
    """collapse_seam_shifted_uv_rows reduces canonical (U' > U) back to OBJ vt structure (U_obj == U): seam-shifted siblings at (u, v) and (u - 1, v) emit one vt line referenced by both face-corner indices.

    Args:
        None.

    Returns:
        None.
    """
    canonical_verts_uvs = torch.tensor(
        [
            [0.97, 0.20],
            [0.02, 0.25],
            [0.05, 0.80],
            [0.10, 0.30],
            [1.02, 0.25],
            [1.05, 0.80],
        ],
        dtype=torch.float32,
    )
    canonical_faces_uvs = torch.tensor([[0, 4, 5], [1, 2, 3]], dtype=torch.int64)

    obj_vt_table, obj_faces_uvs = collapse_seam_shifted_uv_rows(
        verts_uvs=canonical_verts_uvs,
        faces_uvs=canonical_faces_uvs,
    )

    assert obj_vt_table.shape == (4, 2), (
        "Expected the seam-shifted sibling rows to collapse so the vt table "
        "holds U_obj == U rows. "
        f"{obj_vt_table.shape=}"
    )
    for sibling_uv in ([0.02, 0.25], [0.05, 0.80]):
        matches = torch.all(
            torch.isclose(
                obj_vt_table,
                torch.tensor(sibling_uv, dtype=torch.float32).unsqueeze(0),
                atol=1.0e-06,
                rtol=0.0,
            ),
            dim=1,
        )
        assert int(matches.sum().item()) == 1, (
            "Expected the sibling rows at (u, v) and (u - 1, v) to collapse to "
            "one vt entry. "
            f"{sibling_uv=} {obj_vt_table=}"
        )
    assert int(obj_faces_uvs[0, 1]) == int(obj_faces_uvs[1, 0]), (
        "Expected both face-corner indices of the (0.02, 0.25) siblings to "
        "reference the single collapsed vt entry. "
        f"{obj_faces_uvs=}"
    )
    assert int(obj_faces_uvs[0, 2]) == int(obj_faces_uvs[1, 1]), (
        "Expected both face-corner indices of the (0.05, 0.80) siblings to "
        "reference the single collapsed vt entry. "
        f"{obj_faces_uvs=}"
    )
