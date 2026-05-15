"""WebGL rendering utilities for Gaussian-splatting PLY outputs.

Provides backend support for serving 3DGS model data to browser-based
WebGL Gaussian splat renderers. The in-browser renderer
(@mkkellogg/gaussian-splats-3d) loads .ply files natively with full
spherical-harmonics support, so the backend only needs to:

1. inspect trained .ply files for metadata (Gaussian count, SH degree,
   file size) to report to the UI.
2. hand the raw .ply bytes to the Flask route, which is trivial and does
   not need to live here.

Historically this module also produced plotly scatter figures and a
compact .splat binary format. Both are gone: task 20260410_run_3dgs
explicitly forbids plotly for 3DGS and pins the renderer to
GaussianSplats3D, which consumes .ply directly.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from plyfile import PlyData


def load_gaussian_splatting_metadata(
    ply_path: Path,
) -> Dict[str, Any]:
    """Load metadata from a 3DGS .ply file without loading full vertex data.

    Returns a dict with:
        num_gaussians: total number of Gaussians in the model
        sh_degree: spherical harmonics degree (0-3)
        file_size_bytes: size of the .ply file on disk
    """
    assert isinstance(ply_path, Path), f"{type(ply_path)=}"
    assert ply_path.exists(), f"{ply_path=}"

    ply_data = PlyData.read(str(ply_path))
    assert "vertex" in ply_data, f"No vertex element in {ply_path}"
    vertex = ply_data["vertex"]
    vertex_names = list(vertex.data.dtype.names)

    sh_rest_count = sum(1 for n in vertex_names if n.startswith("f_rest_"))
    sh_degree = 0
    if sh_rest_count > 0:
        sh_rest_per_channel = sh_rest_count // 3
        sh_degree = int(np.sqrt(sh_rest_per_channel + 1)) - 1

    return {
        "num_gaussians": int(vertex.count),
        "sh_degree": sh_degree,
        "file_size_bytes": int(ply_path.stat().st_size),
    }
