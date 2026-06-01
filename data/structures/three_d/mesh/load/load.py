from pathlib import Path
from typing import Union

from data.structures.three_d.mesh.load.load_glb import load_glb_mesh
from data.structures.three_d.mesh.load.load_obj import load_obj_mesh
from data.structures.three_d.mesh.mesh import Mesh


def load_mesh(path: Union[str, Path]) -> Mesh:
    """Load one mesh from a GLB file or an OBJ source.

    Dispatches on the path's format: a `.glb` file is loaded as a GLB; an OBJ
    file or a mesh-root directory of OBJs is loaded as one merged OBJ mesh.

    Args:
        path: GLB file path, OBJ file path, or mesh-root directory path.

    Returns:
        One `Mesh`.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )

    _validate_inputs()

    candidate_path = Path(path)
    if candidate_path.is_file() and candidate_path.suffix.lower() == ".glb":
        return load_glb_mesh(path=path)
    if (
        candidate_path.is_file() and candidate_path.suffix.lower() == ".obj"
    ) or candidate_path.is_dir():
        return load_obj_mesh(path=path)
    assert 0, (
        "should not reach here: mesh loading expects a `.glb` file, a `.obj` "
        f"file, or a mesh-root directory. {path=}"
    )
