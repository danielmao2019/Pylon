from typing import Optional, Dict
import os
import numpy as np
import torch
from plyfile import PlyData
import laspy


def _load_from_ply(filename, nameInPly: Optional[str] = None, name_feat: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Read XYZ and optional feature for each vertex from PLY file.

    Args:
        filename: Path to PLY file
        nameInPly: Name of vertex element in PLY (e.g., 'vertex', 'params'). If None, will use first element.
        name_feat: Name of feature column. If None, will return only XYZ coordinates.
    
    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
    """
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)

        # If nameInPly not specified, use first element
        if nameInPly is None:
            nameInPly = plydata.elements[0].name

        num_verts = plydata[nameInPly].count

        # Always read XYZ
        positions = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        positions[:, 0] = plydata[nameInPly].data["x"]
        positions[:, 1] = plydata[nameInPly].data["y"]
        positions[:, 2] = plydata[nameInPly].data["z"]
        
        result = {'pos': positions}

        # Add feature if specified and exists
        if name_feat is not None and name_feat in plydata[nameInPly].data.dtype.names:
            features = plydata[nameInPly].data[name_feat].astype(np.float32).reshape(-1, 1)
            result['feat'] = features

    return result


def _load_from_txt(filename: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from a text file.

    This function handles various text file formats:
    - SLPCCD format: Has header lines and contains XYZ + optional RGB + label
    - General format: Space-separated columns with at least XYZ

    Args:
        filename: Path to the text file

    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
    """
    # SLPCCD format has a header line and point count line that need to be skipped
    # Skip the first two lines (header and point count) for SLPCCD format
    data = np.loadtxt(filename, delimiter=' ', skiprows=2)

    # Extract XYZ coordinates
    positions = data[:, 0:3].astype(np.float32)
    result = {'pos': positions}

    # Handle additional columns as features
    if data.shape[1] > 3:
        # For SLPCCD format, use the 7th column (index 6) if available for label
        if data.shape[1] >= 7:  # X Y Z Rf Gf Bf label
            features = data[:, 6].astype(np.float32).reshape(-1, 1)  # Use label as feature
        else:
            # Otherwise use remaining columns as features
            features = data[:, 3:].astype(np.float32)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
        result['feat'] = features

    return result


def _load_from_las(filename: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from a LAS/LAZ file.

    Args:
        filename: Path to the LAS/LAZ file

    Returns:
        Dictionary containing 'pos' and additional attributes
    """
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
                attr_value = attr_value.reshape(-1, 1)
                result[field] = attr_value

    return result


def _load_from_pth(file_path: str) -> Dict[str, np.ndarray]:
    """Load a point cloud from a PyTorch tensor file (.pth).
    
    Args:
        file_path: Path to the PyTorch tensor file (.pth)
        
    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
    """
    # Load the tensor
    points = torch.load(file_path, map_location='cpu')
    
    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    # Create result dictionary
    result = {'pos': points[:, :3].astype(np.float32)}
    
    # Add features if available
    if points.shape[1] > 3:
        result['feat'] = points[:, 3:].astype(np.float32)
    
    return result


def _load_from_off(filename: str) -> Dict[str, np.ndarray]:
    """Read point cloud data from an OFF file.
    
    Args:
        filename: Path to OFF file
        
    Returns:
        Dictionary with 'pos' containing XYZ coordinates
    """
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError(f"Invalid OFF file format: {filename}")
        
        n_vertices, n_faces, n_edges = map(int, f.readline().strip().split())
        
        vertices = []
        for _ in range(n_vertices):
            line = f.readline().strip()
            coords = list(map(float, line.split()))
            vertices.append(coords[:3])  # Take only XYZ, ignore additional columns
        
        positions = np.array(vertices, dtype=np.float32)
        return {'pos': positions}


def load_point_cloud(
    pathPC,
    nameInPly: Optional[str] = None,
    name_feat: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """Load a point cloud file and return in consistent dictionary format.

    Args:
        pathPC: Path to point cloud file
        nameInPly: Name of vertex element in PLY file (optional)
        name_feat: Name of feature column (optional)

    Returns:
        Dictionary with at least 'pos' key containing XYZ coordinates as torch.float32
        Additional keys may include 'feat', 'rgb', etc. depending on file format
    """
    pathPC = os.path.normpath(pathPC).replace('\\', '/')

    # Check file existence once at the beginning
    if not os.path.isfile(pathPC):
        raise FileNotFoundError(f"Point cloud file not found: {pathPC}")

    file_ext = os.path.splitext(pathPC)[1].lower()

    # Load data using appropriate loader
    if file_ext == '.pth':
        pc_data = _load_from_pth(pathPC)
    elif file_ext == '.ply':
        pc_data = _load_from_ply(pathPC, nameInPly=nameInPly, name_feat=name_feat)
    elif file_ext in ['.las', '.laz']:
        pc_data = _load_from_las(pathPC)
    elif file_ext == '.off':
        pc_data = _load_from_off(pathPC)
    elif file_ext == '.txt':
        pc_data = _load_from_txt(pathPC)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # All loaders now return dict format with numpy arrays
    # Convert all numpy arrays to torch tensors
    result = {}
    for key, value in pc_data.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            # Ensure pos field is float32
            if key == 'pos':
                tensor = tensor.float()
            result[key] = tensor
        else:
            result[key] = value
    
    # Validate that we have at least position data
    if 'pos' not in result:
        raise ValueError(f"Point cloud data must contain 'pos' field")
    
    # Validate position shape
    pos = result['pos']
    if pos.ndim != 2 or pos.shape[1] < 3:
        raise ValueError(f"Position data must have shape (N, D) where D >= 3, got {pos.shape}")
    
    # Handle segmentation files - convert labels to int64
    is_seg_file = '_seg' in os.path.basename(pathPC).lower()
    if is_seg_file and 'feat' in result:
        result['feat'] = result['feat'].long()
    
    return result
