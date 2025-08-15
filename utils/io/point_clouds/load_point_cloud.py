from typing import Optional, Dict, Union
import os
import numpy as np
import torch
from plyfile import PlyData
import laspy
from utils.input_checks.point_cloud import check_point_cloud


def _load_from_ply(filepath, nameInPly: Optional[str] = None, name_feat: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Read XYZ and optional feature for each vertex from PLY file.

    Args:
        filename: Path to PLY file
        nameInPly: Name of vertex element in PLY (e.g., 'vertex', 'params'). If None, will use first element.
        name_feat: Name of feature column. If None, will return only XYZ coordinates.
    
    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
        All data loaded in float64 precision for maximum accuracy.
    """
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)

        # If nameInPly not specified, use first element
        if nameInPly is None:
            nameInPly = plydata.elements[0].name

        num_verts = plydata[nameInPly].count

        # Always read XYZ in float64 precision
        positions = np.zeros(shape=[num_verts, 3], dtype=np.float64)
        positions[:, 0] = plydata[nameInPly].data["x"].astype(np.float64)
        positions[:, 1] = plydata[nameInPly].data["y"].astype(np.float64)
        positions[:, 2] = plydata[nameInPly].data["z"].astype(np.float64)
        
        result = {'pos': positions}

        # Add RGB colors if available
        rgb_fields = ['red', 'green', 'blue']
        if all(field in plydata[nameInPly].data.dtype.names for field in rgb_fields):
            # Extract RGB values and normalize to [0, 1] range - use float64 for precision
            red = plydata[nameInPly].data["red"].astype(np.float64) / 255.0
            green = plydata[nameInPly].data["green"].astype(np.float64) / 255.0
            blue = plydata[nameInPly].data["blue"].astype(np.float64) / 255.0
            rgb = np.column_stack((red, green, blue))
            result['rgb'] = rgb

        # Add feature if specified and exists
        if name_feat is not None and name_feat in plydata[nameInPly].data.dtype.names:
            features = plydata[nameInPly].data[name_feat].astype(np.float64).reshape(-1, 1)
            result['feat'] = features

    return result


def _load_from_txt(filepath: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from a text file.

    Args:
        filepath: Path to the text file

    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
        All data loaded in float64 precision for maximum accuracy.
    """
    # Load data in float64 precision - SLPCCD format has header lines that need to be skipped
    data = np.loadtxt(filepath, delimiter=' ', skiprows=2, dtype=np.float64)

    # Extract XYZ coordinates
    positions = data[:, 0:3]
    result = {'pos': positions}

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
        Dictionary containing 'pos' and additional attributes
        All data loaded in float64 precision for maximum accuracy.
    """
    # Read the LAS/LAZ file
    las_file = laspy.read(filepath)

    # Extract XYZ coordinates in float64 precision
    points = np.vstack((
        np.array(las_file.x, dtype=np.float64), 
        np.array(las_file.y, dtype=np.float64), 
        np.array(las_file.z, dtype=np.float64)
    )).T

    # Initialize result dictionary with position
    result = {'pos': points}

    # Extract RGB colors if available
    if all(field in las_file.point_format.dimension_names for field in ['red', 'green', 'blue']):
        # Normalize RGB values to [0, 1] range - use float64 for precision
        red_array = np.array(las_file.red, dtype=np.float64)
        green_array = np.array(las_file.green, dtype=np.float64)
        blue_array = np.array(las_file.blue, dtype=np.float64)
        
        red = red_array / np.max(red_array)
        green = green_array / np.max(green_array)
        blue = blue_array / np.max(blue_array)
        rgb = np.vstack((red, green, blue)).T
        result['rgb'] = rgb

    # Add all available attributes
    for field in las_file.point_format.dimension_names:
        if field not in ['x', 'y', 'z', 'red', 'green', 'blue']:  # Skip XYZ and RGB as they're already handled
            attr_value = getattr(las_file, field)
            if attr_value is not None:
                # Keep the original dtype - don't force conversion to float64
                attr_value = np.array(attr_value).reshape(-1, 1)
                result[field] = attr_value

    return result


def _load_from_pth(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Load a point cloud from a PyTorch tensor file (.pth).
    
    Args:
        filepath: Path to the PyTorch tensor file (.pth)
        device: Device parameter (ignored - kept for API consistency)
        
    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features.
        Returns data in whatever format was saved (torch.Tensor or np.ndarray).
    """
    # Load the data - can be either torch.Tensor or np.ndarray
    data = torch.load(filepath, map_location='cpu')
    
    # Handle both torch.Tensor and np.ndarray
    assert isinstance(data, (torch.Tensor, np.ndarray))
    result = {'pos': data[:, :3]}
    if data.shape[1] > 3:
        result['feat'] = data[:, 3:]
    
    return result


def _load_from_off(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, torch.Tensor]:
    """Read point cloud data from an OFF file.
    
    Args:
        filepath: Path to OFF file
        
    Returns:
        Dictionary with 'pos' containing XYZ coordinates
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError(f"Invalid OFF file format: {filepath}")
        
        n_vertices, _, _ = map(int, f.readline().strip().split())
        
        vertices = []
        for _ in range(n_vertices):
            line = f.readline().strip()
            coords = list(map(float, line.split()))
            vertices.append(coords[:3])  # Take only XYZ, ignore additional columns
        
        positions = torch.tensor(vertices, dtype=torch.float32, device=device)
        return {'pos': positions}


def load_point_cloud(
    filepath,
    nameInPly: Optional[str] = None,
    name_feat: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """Load a point cloud file and return in consistent dictionary format.

    Args:
        filepath: Path to point cloud file
        nameInPly: Name of vertex element in PLY file (optional)
        name_feat: Name of feature column (optional)
        device: Device to place tensors on ('cuda', 'cpu', or torch.device)
        dtype: Precision for position data (torch.float32 or torch.float64)
               Helper loaders always use float64 internally, then convert to requested dtype

    Returns:
        Dictionary with at least 'pos' key containing XYZ coordinates in requested dtype
        Additional keys may include 'feat', 'rgb', etc. depending on file format
    """
    filepath = os.path.normpath(filepath).replace('\\', '/')

    # Check file existence once at the beginning
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    file_ext = os.path.splitext(filepath)[1].lower()

    # Load data using appropriate loader
    if file_ext == '.pth':
        pc_data = _load_from_pth(filepath, device=device)
    elif file_ext == '.ply':
        pc_data = _load_from_ply(filepath, nameInPly=nameInPly, name_feat=name_feat)
    elif file_ext in ['.las', '.laz']:
        pc_data = _load_from_las(filepath)
    elif file_ext == '.off':
        pc_data = _load_from_off(filepath, device=device)
    elif file_ext == '.txt':
        pc_data = _load_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Convert any numpy arrays to torch tensors and transfer to device
    def numpy_to_torch_on_device(key, x):
        if isinstance(x, np.ndarray):
            if x.dtype == np.uint16:
                x = x.astype(np.int64)
            tensor = torch.from_numpy(x).to(device)
            # Apply requested dtype only to position data
            if key == 'pos':
                tensor = tensor.to(dtype)
            return tensor
        elif isinstance(x, torch.Tensor):
            tensor = x.to(device)
            # Apply requested dtype only to position data
            if key == 'pos':
                tensor = tensor.to(dtype)
            return tensor
        else:
            return x
    
    # Apply numpy to torch conversion and device transfer
    result = {key: numpy_to_torch_on_device(key, value) for key, value in pc_data.items()}
    
    # Ensure pos exists and is in correct dtype
    assert 'pos' in result
    
    # Handle segmentation files - convert labels to int64
    is_seg_file = '_seg' in os.path.basename(filepath).lower()
    if is_seg_file and 'feat' in result:
        result['feat'] = result['feat'].long()
    
    # Validate result using input checks
    check_point_cloud(result)
    
    return result
