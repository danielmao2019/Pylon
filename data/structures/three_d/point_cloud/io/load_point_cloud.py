import os
from typing import Dict, Optional, Union

import laspy
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from torch.utils import dlpack as torch_dlpack

from data.structures.three_d.point_cloud.point_cloud import PointCloud


def _load_from_ply(
    filepath: str,
    nameInPly: Optional[str] = None,
    name_feat: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Read XYZ and all available fields from PLY file.

    Args:
        filename: Path to PLY file
        nameInPly: Name of vertex element in PLY (e.g., 'vertex', 'params'). If None, will use first element.
        name_feat: Name of feature column (deprecated - all fields are now loaded automatically).

    Returns:
        Dictionary with 'xyz' containing coordinates and all other available fields
        All data loaded preserving original precision where possible.
    """
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)

        # If nameInPly not specified, use first element
        if nameInPly is None:
            assert (
                len(plydata.elements) == 1
            ), f"PLY file must have exactly one element, got: {list(plydata.elements.keys())}"
            nameInPly = plydata.elements[0].name

        num_verts = plydata[nameInPly].count
        available_fields = plydata[nameInPly].data.dtype.names

        # Always read XYZ in float64 precision
        positions = np.zeros(shape=[num_verts, 3], dtype=np.float64)
        positions[:, 0] = plydata[nameInPly].data["x"].astype(np.float64)
        positions[:, 1] = plydata[nameInPly].data["y"].astype(np.float64)
        positions[:, 2] = plydata[nameInPly].data["z"].astype(np.float64)

        result = {'xyz': positions}

        # Add RGB colors if available - preserve original data types and values
        rgb_fields = ['red', 'green', 'blue']
        if all(field in available_fields for field in rgb_fields):
            # Extract RGB values preserving original data type and values (no normalization)
            red = plydata[nameInPly].data["red"]
            green = plydata[nameInPly].data["green"]
            blue = plydata[nameInPly].data["blue"]
            rgb = np.column_stack((red, green, blue))
            result['rgb'] = np.ascontiguousarray(rgb)

        # Load ALL other fields dynamically (except x, y, z, red, green, blue)
        processed_fields = {'x', 'y', 'z', 'red', 'green', 'blue'}
        for field_name in available_fields:
            if field_name not in processed_fields:
                field_data = plydata[nameInPly].data[field_name]

                # STEP 1: Load as-is, preserving original shape and dtype
                field_array = np.ascontiguousarray(field_data)

                # STEP 2: Check if shape is [N, 1] and squeeze if needed
                if field_array.ndim == 2 and field_array.shape[1] == 1:
                    field_array = field_array.squeeze(axis=1)

                result[field_name] = field_array

        # Add feature if specified and exists (legacy compatibility)
        if (
            name_feat is not None
            and name_feat in available_fields
            and name_feat not in result
        ):
            features = (
                plydata[nameInPly].data[name_feat].astype(np.float64).reshape(-1, 1)
            )
            result['feat'] = features

    return result


def _load_from_txt(filepath: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from a text file.

    Args:
        filepath: Path to the text file

    Returns:
        Dictionary with 'xyz' containing coordinates and optional 'feat' for additional features
        All data loaded in float64 precision for maximum accuracy.
    """
    # Load data in float64 precision - SLPCCD format has header lines that need to be skipped
    data = np.loadtxt(filepath, delimiter=' ', skiprows=2, dtype=np.float64)

    # Extract XYZ coordinates
    positions = data[:, 0:3]
    result = {'xyz': positions}

    # Extract features if available
    if data.shape[1] > 3:
        if data.shape[1] >= 7:
            # SLPCCD format: X Y Z Rf Gf Bf label - use label column as feature
            features = data[:, 6:7]
        else:
            # General format: use all remaining columns as features
            features = data[:, 3:]

        result['feat'] = features

    return result


def _load_from_las(filepath: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from a LAS/LAZ file.

    Args:
        filename: Path to the LAS/LAZ file

    Returns:
        Dictionary containing 'xyz' and additional attributes
        All data loaded in float64 precision for maximum accuracy.
    """
    # Read the LAS/LAZ file
    las_file = laspy.read(filepath)

    # Extract XYZ coordinates in float64 precision
    points = np.vstack(
        (
            np.array(las_file.x, dtype=np.float64),
            np.array(las_file.y, dtype=np.float64),
            np.array(las_file.z, dtype=np.float64),
        )
    ).T

    # Initialize result dictionary with position
    result = {'xyz': points}

    # Extract RGB colors if available - preserve original values and data types
    if all(
        field in las_file.point_format.dimension_names
        for field in ['red', 'green', 'blue']
    ):
        # Keep original RGB values without normalization
        red_array = np.array(las_file.red)
        green_array = np.array(las_file.green)
        blue_array = np.array(las_file.blue)

        rgb = np.vstack((red_array, green_array, blue_array)).T
        result['rgb'] = rgb

    # Add all available attributes
    for field in las_file.point_format.dimension_names:
        if field not in [
            'x',
            'y',
            'z',
            'red',
            'green',
            'blue',
        ]:  # Skip XYZ and RGB as they're already handled
            attr_value = getattr(las_file, field)
            if attr_value is not None:
                # STEP 1: Load as-is, preserving original shape and dtype
                attr_value = np.array(attr_value)

                # STEP 2: Check if shape is [N, 1] and squeeze if needed
                if attr_value.ndim == 2 and attr_value.shape[1] == 1:
                    attr_value = attr_value.squeeze(axis=1)

                result[field] = attr_value

    return result


def _load_from_pcd(filepath: str) -> Dict[str, torch.Tensor]:
    """Read point cloud data from a PCD file using Open3D tensor IO.

    Returns Dict[str, torch.Tensor] directly without dtype/device conversions.
    """

    tensor_pcd = o3d.t.io.read_point_cloud(filepath)

    assert (
        'positions' in tensor_pcd.point
    ), f"PCD file does not contain positions: {filepath}"

    # Return Open3D tensors converted to torch.Tensor directly, as-is
    pos_t: torch.Tensor = torch_dlpack.from_dlpack(
        tensor_pcd.point['positions'].to_dlpack()
    )
    result: Dict[str, torch.Tensor] = {'xyz': pos_t}

    if 'colors' in tensor_pcd.point:
        result['rgb'] = torch_dlpack.from_dlpack(tensor_pcd.point['colors'].to_dlpack())

    for field_name, ten in tensor_pcd.point.items():
        if field_name in {'positions', 'colors'}:
            continue
        # Keep original shape and dtype; no squeezing or casting
        result[field_name] = torch_dlpack.from_dlpack(ten.to_dlpack())

    return result


def _load_from_pth(
    filepath: str, device: Union[str, torch.device] = 'cuda'
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Load a point cloud from a PyTorch tensor file (.pth).

    Args:
        filepath: Path to the PyTorch tensor file (.pth)
        device: Device parameter (ignored - kept for API consistency)

    Returns:
        Dictionary with 'xyz' containing coordinates and optional 'feat' for additional features.
        Returns data in whatever format was saved (torch.Tensor or np.ndarray).
    """
    # Load the data - can be either torch.Tensor or np.ndarray
    data = torch.load(filepath, map_location='cpu')

    # Handle both torch.Tensor and np.ndarray
    assert isinstance(data, (torch.Tensor, np.ndarray))
    result = {'xyz': data[:, :3]}
    if data.shape[1] > 3:
        result['feat'] = data[:, 3:]

    return result


def _load_from_off(
    filepath: str, device: Union[str, torch.device] = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Read point cloud data from an OFF file.

    Args:
        filepath: Path to OFF file

    Returns:
        Dictionary with 'xyz' containing coordinates
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        assert header == 'OFF', f"Invalid OFF file format: {filepath}"

        n_vertices, _, _ = map(int, f.readline().strip().split())

        vertices = []
        for _ in range(n_vertices):
            line = f.readline().strip()
            coords = list(map(float, line.split()))
            vertices.append(coords[:3])  # Take only XYZ, ignore additional columns

        positions = torch.tensor(vertices, dtype=torch.float32, device=device)
        return {'xyz': positions}


def load_point_cloud(
    filepath: str,
    nameInPly: Optional[str] = None,
    name_feat: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> PointCloud:
    """Load a point cloud file and return as PointCloud.

    Args:
        filepath: Path to point cloud file
        nameInPly: Name of vertex element in PLY file (optional)
        name_feat: Name of feature column (optional)
        device: Device to place tensors on ('cuda', 'cpu', or torch.device)
        dtype: Precision for position data (torch.float32 or torch.float64)

    Returns:
        PointCloud with coordinates in requested dtype.
        Additional fields may include 'feat', 'rgb', etc. depending on file format.
    """

    def _validate_inputs() -> None:
        assert os.path.splitext(filepath)[1] in {
            '.pth',
            '.ply',
            '.pcd',
            '.las',
            '.laz',
            '.off',
            '.txt',
        }, f"Unsupported file format: {os.path.splitext(filepath)[1]} from {filepath=}"

    _validate_inputs()

    def _normalize_inputs(filepath: str) -> str:
        filepath = os.path.normpath(filepath).replace('\\', '/')
        assert os.path.isfile(filepath), f"Point cloud file not found: {filepath}"
        return filepath

    filepath = _normalize_inputs(filepath=filepath)

    def _load_by_format() -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Read the file through the one reader that owns its extension.

        Args:
            None. The enclosing call's filepath, nameInPly, name_feat and device are read from the closure.

        Returns:
            Dictionary keyed by field name ('xyz' plus whatever else the format carries), holding the arrays or tensors the matching reader produced.
        """
        file_ext = os.path.splitext(filepath)[1]
        if file_ext == '.pth':
            return _load_from_pth(filepath, device=device)
        if file_ext == '.ply':
            return _load_from_ply(filepath, nameInPly=nameInPly, name_feat=name_feat)
        if file_ext == '.pcd':
            return _load_from_pcd(filepath)
        if file_ext in ['.las', '.laz']:
            return _load_from_las(filepath)
        if file_ext == '.off':
            return _load_from_off(filepath, device=device)
        if file_ext == '.txt':
            return _load_from_txt(filepath)
        assert 0, "Should not reach here."

    pc_data = _load_by_format()

    def _normalize_field(key: str, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Place one loaded field on the requested device and cast it to the dtype its key calls for.

        Args:
            key: Field name the matched reader returned this field under.
            x: The field exactly as the reader produced it, an np.ndarray or a torch.Tensor.

        Returns:
            The field as a torch.Tensor on device, positions cast to dtype and a segmentation
            file's label column cast to torch.int64.
        """
        if isinstance(x, np.ndarray):
            if x.dtype == np.uint16:
                x = x.astype(np.int32)
            x = torch.from_numpy(x)

        assert isinstance(x, torch.Tensor), f"{type(x)=} under {key=}"
        tensor = x.to(device)

        if key == 'xyz':
            assert tensor.is_floating_point(), f"{tensor.dtype=} from {filepath=}"
            tensor = tensor.to(dtype)

        is_seg_file = '_seg' in os.path.basename(filepath)
        if key == 'feat' and is_seg_file:
            assert tensor.is_floating_point() or tensor.dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ), f"{tensor.dtype=} from {filepath=}"
            tensor = tensor.to(torch.int64)

        return tensor

    result = {}
    for key, value in pc_data.items():
        result[key] = _normalize_field(key, value)

    return PointCloud(data=result)
