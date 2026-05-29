from pathlib import Path
from typing import Union

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.save.save_glb import save_glb_mesh
from data.structures.three_d.mesh.save.save_obj import save_obj_mesh
from data.structures.three_d.mesh.save.save_ply import save_ply_mesh


def save_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None:
    """Save one mesh to OBJ, PLY, or GLB, dispatched on the output path's format.

    OBJ vs PLY vs GLB is the top-level responsibility split; a directory-like
    path (no suffix) defaults to an OBJ file.

    Args:
        mesh: `Mesh` instance to save.
        output_path: Output mesh filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    suffix = Path(output_path).suffix.lower()
    if suffix == ".glb":
        save_glb_mesh(mesh=mesh, output_path=output_path)
        return
    if suffix == ".ply":
        save_ply_mesh(mesh=mesh, output_path=output_path)
        return
    if suffix == ".obj" or suffix == "":
        save_obj_mesh(mesh=mesh, output_path=output_path)
        return
    assert 0, (
        "should not reach here: mesh saving expects a `.glb`, `.ply`, or `.obj` "
        f"file, or a directory-like path. {output_path=}"
    )
