import os
from typing import Any, Dict, Optional, Union

import laspy
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData

from data.structures.three_d.point_cloud.point_cloud import PointCloud


def load_point_cloud(
    filepath: str,
    meta_data: Optional[Dict[str, Dict[str, Any]]] = None,
    device: Union[str, torch.device] = 'cuda',
) -> PointCloud:
    """Loads one point cloud file of any supported format as the cloud its own columns define, then applies the meta data over the halves that source leaves for the caller.

    Args:
        filepath: The path of the point cloud file to read, as a str carrying one of the extensions '.pth', '.ply', '.pcd', '.las', '.laz', '.off' and '.txt'.
        meta_data: The override, one entry per field keyed by field name, each entry stating a 'dtype' naming a conceptual dtype, a 'layout' naming source columns as a tuple of str, or both, or None to apply the source's own record with nothing written over it.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The loaded point cloud, as a PointCloud carrying an xyz field of shape [N, 3] beside every other field the source's columns and the override assemble.
    """

    def _validate_inputs() -> None:
        assert os.path.splitext(filepath)[1] in (
            '.pth',
            '.ply',
            '.pcd',
            '.las',
            '.laz',
            '.off',
            '.txt',
        ), f"a file is read by the reader that owns its extension, and no reader owns this one: filepath={filepath}, extension={os.path.splitext(filepath)[1]}"

    _validate_inputs()

    def _normalize_inputs(filepath: str) -> str:
        filepath = filepath.replace('\\', '/')
        # output validation of the rewrite: the normalized path is the one that has to exist
        assert os.path.isfile(
            filepath
        ), f"a point cloud is read from a file that exists: filepath={filepath}"
        return filepath

    filepath = _normalize_inputs(filepath=filepath)

    pc = _load_by_format(filepath=filepath, device=device)
    pc.apply_meta_data(meta_data=meta_data)
    # a raw cloud without coordinates is legal, a loaded one is not, so this is where a positional source that named no layout aborts
    assert (
        'xyz' in pc.field_names()
    ), f"a loaded cloud carries coordinates, so a source numbering its columns is loaded under a layout naming which three are the coordinates: filepath={filepath}, fields={pc.field_names()}, meta_data={meta_data}"

    return pc


def _load_by_format(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads the file through the one reader that owns its extension.

    Args:
        filepath: The path of the point cloud file to read, as a str carrying one of the supported extensions.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud the matching reader built, as a PointCloud whose fields are the source's own columns.
    """
    file_ext = os.path.splitext(filepath)[1]
    if file_ext == '.pth':
        return _load_from_pth(filepath, device)
    if file_ext == '.ply':
        return _load_from_ply(filepath, device)
    if file_ext == '.pcd':
        return _load_from_pcd(filepath, device)
    if file_ext in ['.las', '.laz']:
        return _load_from_las(filepath, device)
    if file_ext == '.off':
        return _load_from_off(filepath, device)
    if file_ext == '.txt':
        return _load_from_txt(filepath, device)
    assert 0, "Should not reach here."


def _load_from_pth(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads a .pth file holding one block the file names nothing about, each column becoming a field under its own index.

    Args:
        filepath: The path of the .pth file to read, as a str.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud it built, as a PointCloud whose fields are the block's columns keyed by index, its coordinates unnamed until the caller's meta data names them.
    """
    data = torch.load(filepath, map_location='cpu')
    assert isinstance(
        data, (torch.Tensor, np.ndarray)
    ), f"a .pth holds one block of values, as a torch tensor or a numpy array: filepath={filepath}, type(data)={type(data)}"
    # a block of one axis has no columns to key by index, and splitting one raises an IndexError from inside the split rather than refusing the file at this door
    assert (
        data.ndim == 2
    ), f"a .pth block carries a column axis to key its columns by: filepath={filepath}, data.shape={tuple(data.shape)}"
    columns = {str(index): data[:, index] for index in range(data.shape[1])}

    return PointCloud(data=columns, device=device)


def _load_from_ply(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads a PLY's properties as fields, each in the dtype the file stores it in.

    Args:
        filepath: The path of the .ply file to read, as a str.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud it built, as a PointCloud whose fields are the file's properties, a multi-element file qualifying every key so PLY's own coordinate names are absent from it.
    """
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
        assert (
            len(plydata.elements) >= 1
        ), f"a PLY carries at least one element to read columns off: filepath={filepath}, elements={tuple(element.name for element in plydata.elements)}"
        columns = {
            (
                property_name
                if len(plydata.elements) == 1
                else f"{element.name}.{property_name}"
            ): np.ascontiguousarray(element.data[property_name])
            for element in plydata.elements
            for property_name in element.data.dtype.names
        }

        return PointCloud(data=columns, device=device)


def _load_from_pcd(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads a PCD through Open3D's tensor IO, each attribute becoming a field whole under its own name.

    Args:
        filepath: The path of the .pcd file to read, as a str.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud it built, as a PointCloud whose fields are the file's Open3D attributes.
    """
    tensor_pcd = o3d.t.io.read_point_cloud(filepath)
    columns = {}
    # a TensorMap iterates its names alone, so the pairs come from items rather than from the map itself
    for attribute_name, ten in tensor_pcd.point.items():
        # an attribute is one named block, the way an in-memory variable is, so it is not split into columns of its own
        columns[attribute_name] = ten.numpy()

    return PointCloud(data=columns, device=device)


def _load_from_las(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads a LAS/LAZ file's dimensions as fields, each in the dtype laspy materializes it as, which makes a bit-packed dimension an ordinary uint8 field.

    Args:
        filepath: The path of the .las or .laz file to read, as a str.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud it built, as a PointCloud whose fields are the file's dimensions beside the scaled real-world coordinate columns 'x', 'y' and 'z'.
    """
    las_file = laspy.read(filepath)
    columns = {}
    for dimension_name in las_file.point_format.dimension_names:
        if dimension_name not in ('X', 'Y', 'Z'):
            columns[dimension_name] = np.asarray(getattr(las_file, dimension_name))
    columns['x'], columns['y'], columns['z'] = (
        np.asarray(las_file.x),
        np.asarray(las_file.y),
        np.asarray(las_file.z),
    )

    return PointCloud(data=columns, device=device)


def _load_from_off(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads the vertex block of an OFF file into float32 coordinate fields.

    Args:
        filepath: The path of the .off file to read, as a str.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud it built, as a PointCloud carrying the float32 coordinate fields 'x', 'y' and 'z'.
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        # ModelNet40 writes the counts glued to the keyword, so OFF is the line's prefix rather than the whole of it
        assert header.startswith(
            'OFF'
        ), f"an OFF file opens on the OFF keyword: filepath={filepath}, header={header}"
        counts_text = header[len('OFF') :].strip() or next(
            line for line in f if line.strip() and not line.strip().startswith('#')
        )
        n_vertices = int(counts_text.split()[0])
        vertices = []
        for _ in range(n_vertices):
            coords = list(map(float, f.readline().strip().split()))
            vertices.append(coords[:3])
        # float32 is the width this format is READ at, so the text lands there directly rather than being parsed wide and narrowed onto float32's grid afterwards, which every ordinary decimal would fail
        positions = np.array(vertices, dtype=np.float32)
        # a magnitude beyond float32 overflows in the parse, and validate_xyz_tensor is what aborts on it, this reader not repeating a check the construction it feeds already makes
        columns = {
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
        }

        return PointCloud(data=columns, device=device)


def _load_from_txt(filepath: str, device: Union[str, torch.device]) -> PointCloud:
    """Reads a whitespace-separated text point cloud whose leading two lines are a header, each column becoming a field under its own index and nothing divined from how many there are.

    Args:
        filepath: The path of the .txt file to read, as a str, its leading two lines a header.
        device: The torch device every field is to sit on, as a str or a torch.device.

    Returns:
        The raw cloud it built, as a PointCloud whose fields are the table's columns keyed by index, its coordinates unnamed until the caller's meta data names them.
    """
    # no delimiter is named because the default splits on runs of whitespace, taking aligned columns and the single-space file alike, and ndmin holds the column axis open on the one-row file numpy would otherwise hand back flat
    data = np.loadtxt(filepath, skiprows=2, dtype=np.float64, ndmin=2)
    columns = {str(index): data[:, index] for index in range(data.shape[1])}

    return PointCloud(data=columns, device=device)
