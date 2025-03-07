from typing import Optional
import os
import numpy as np
import torch
from plyfile import PlyData


def _read_from_ply(filename, nameInPly: str, name_feat: str) -> np.ndarray:
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata[nameInPly].count
        vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
        vertices[:, 0] = plydata[nameInPly].data["x"]
        vertices[:, 1] = plydata[nameInPly].data["y"]
        vertices[:, 2] = plydata[nameInPly].data["z"]
        vertices[:, 3] = plydata[nameInPly].data[name_feat]
    return vertices


def load_point_cloud(pathPC, nameInPly: Optional[str] = None, name_feat: Optional[str] = "label_ch") -> torch.Tensor:
    """
    load a tile and returns points features (normalized xyz + intensity) and
    ground truth
    INPUT:
    pathPC = string, path to the tile of PC
    OUTPUT
    pc_data, [n x 3] float array containing points coordinates and intensity
    lbs, [n] long int array, containing the points semantic labels
    """
    pc_data = _read_from_ply(pathPC, nameInPly="params" if nameInPly is None else nameInPly, name_feat=name_feat)
    pc_data = torch.from_numpy(pc_data)
    return pc_data
