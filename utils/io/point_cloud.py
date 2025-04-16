from typing import Optional, Dict, Union
import os
import numpy as np
import torch
from plyfile import PlyData
import laspy


def _read_from_ply(filename, nameInPly: Optional[str] = None, name_feat: Optional[str] = None) -> np.ndarray:
    """Read XYZ and optional feature for each vertex.

    Args:
        filename: Path to PLY file
        nameInPly: Name of vertex element in PLY (e.g., 'vertex', 'params'). If None, will use first element.
        name_feat: Name of feature column. If None, will return only XYZ coordinates.
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)

        # If nameInPly not specified, use first element
        if nameInPly is None:
            nameInPly = plydata.elements[0].name

        num_verts = plydata[nameInPly].count

        # Always read XYZ
        vertices = np.zeros(shape=[num_verts, 3 if name_feat is None else 4], dtype=np.float32)
        vertices[:, 0] = plydata[nameInPly].data["x"]
        vertices[:, 1] = plydata[nameInPly].data["y"]
        vertices[:, 2] = plydata[nameInPly].data["z"]

        # Add feature if specified and exists
        if name_feat is not None and name_feat in plydata[nameInPly].data.dtype.names:
            vertices[:, 3] = plydata[nameInPly].data[name_feat]

    return vertices


def _read_from_txt(filename: str) -> np.ndarray:
    """Read point cloud data from a text file.

    This function is specialized for the SLPCCD dataset text file format.
    Point cloud text files typically contain space-separated columns:
    - First 3 columns: XYZ coordinates
    - Optional 4th column: intensity/color/label
    - Additional columns may contain RGB values or other features

    Args:
        filename: Path to the text file

    Returns:
        A numpy array of shape [num_points, 4] containing:
        - XYZ coordinates in columns 0-2
        - Label/intensity in column 3 (0.0 if not present in the file)
    """
    assert os.path.isfile(filename), f"File not found: {filename}"

    # SLPCCD format has a header line and point count line that need to be skipped
    try:
        # Skip the first two lines (header and point count) for SLPCCD format
        data = np.loadtxt(filename, delimiter=' ', skiprows=2)

        # Validate data has at least XYZ coordinates
        if data.shape[1] < 3:
            raise ValueError(f"Point cloud file has less than 3 dimensions: {filename}")

        # Create output array with XYZ + 1 feature channel
        vertices = np.zeros(shape=[data.shape[0], 4], dtype=np.float32)

        # Copy XYZ coordinates
        vertices[:, 0:3] = data[:, 0:3]

        # For SLPCCD format, use the 7th column (index 6) if available for label
        if data.shape[1] >= 7:  # X Y Z Rf Gf Bf label
            vertices[:, 3] = data[:, 6]  # Use label as feature
        elif data.shape[1] >= 4:
            # Otherwise use 4th column as feature
            vertices[:, 3] = data[:, 3]
        else:
            # Set to 0 if not available
            vertices[:, 3] = 0.0

        return vertices
    except Exception as e:
        raise IOError(f"Failed to load point cloud from {filename}: {str(e)}")


def _read_from_las(filename: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from a LAS/LAZ file.

    This function extracts XYZ coordinates and all available attributes from LAS/LAZ files.

    Args:
        filename: Path to the LAS/LAZ file

    Returns:
        A dictionary containing:
        - 'pos': XYZ coordinates in shape [N, 3]
        - 'rgb': RGB colors in shape [N, 3] if available
        - Additional fields for each available attribute in shape [N, 1]
    """
    assert os.path.isfile(filename), f"File not found: {filename}"

    # Read the LAS/LAZ file
    las_file = laspy.read(filename)

    # Extract XYZ coordinates
    points = np.vstack((las_file.x, las_file.y, las_file.z)).T.astype(np.float32)

    # Initialize result dictionary with position
    result = {'pos': points}

    # Extract RGB colors if available
    if all(field in las_file.point_format.dimension_names for field in ['red', 'green', 'blue']):
        # Normalize RGB values to [0, 1] range
        red = las_file.red / np.max(las_file.red)
        green = las_file.green / np.max(las_file.green)
        blue = las_file.blue / np.max(las_file.blue)
        rgb = np.vstack((red, green, blue)).T.astype(np.float32)
        result['rgb'] = rgb

    # Add all available attributes
    for field in las_file.point_format.dimension_names:
        if field not in ['x', 'y', 'z', 'red', 'green', 'blue']:  # Skip XYZ and RGB as they're already handled
            attr_value = getattr(las_file, field)
            if attr_value is not None:
                attr_value = np.array(attr_value, dtype=np.float32)
                assert attr_value.shape == (len(points),), f"{attr_value.shape=}"
                attr_value = attr_value.reshape(-1, 1)
                result[field] = attr_value

    return result


def load_point_cloud(
    pathPC,
    nameInPly: Optional[str] = None,
    name_feat: Optional[str] = None
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Load a point cloud file.

    Args:
        pathPC: Path to point cloud file
        nameInPly: Name of vertex element in PLY file (optional)
        name_feat: Name of feature column (optional)

    Returns:
        Tensor of shape [N, 3] or [N, 4] containing XYZ coordinates and optional feature
    """
    pathPC = os.path.normpath(pathPC).replace('\\', '/')

    if not os.path.isfile(pathPC):
        raise FileNotFoundError(f"Point cloud file not found: {pathPC}")

    file_ext = os.path.splitext(pathPC)[1].lower()

    if file_ext == '.ply':
        pc_data = _read_from_ply(pathPC, nameInPly=nameInPly, name_feat=name_feat)
        pc_data = torch.from_numpy(pc_data)
    elif file_ext in ['.las', '.laz']:
        pc_data = _read_from_las(pathPC)
        assert isinstance(pc_data, dict)
        pc_data = {
            key: torch.from_numpy(val)
            for key, val in pc_data.items()
        }
    else:
        # Check if this is a segmentation file (_seg.txt) for SLPCCD dataset
        is_seg_file = '_seg' in os.path.basename(pathPC).lower()
        pc_data = _read_from_txt(pathPC)

        # For segmentation files, make sure the 4th column (labels) is correctly loaded
        # and converted to integer values for classification
        if is_seg_file:
            pc_data[:, 3] = pc_data[:, 3].astype(np.int64)
        pc_data = torch.from_numpy(pc_data)

    return pc_data
