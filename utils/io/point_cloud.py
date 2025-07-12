from typing import Optional, Dict, Union
import os
import numpy as np
import torch
from plyfile import PlyData
import laspy
from utils.input_checks.point_cloud import check_point_cloud
from utils.ops.apply import apply_tensor_op


def _load_from_ply(filepath, nameInPly: Optional[str] = None, name_feat: Optional[str] = None, device: Union[str, torch.device] = 'cuda') -> Dict[str, torch.Tensor]:
    """Read XYZ and optional feature for each vertex from PLY file.

    Args:
        filename: Path to PLY file
        nameInPly: Name of vertex element in PLY (e.g., 'vertex', 'params'). If None, will use first element.
        name_feat: Name of feature column. If None, will return only XYZ coordinates.
    
    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
    """
    with open(filepath, "rb") as f:
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
        
        result = {'pos': torch.from_numpy(positions).to(device)}

        # Add feature if specified and exists
        if name_feat is not None and name_feat in plydata[nameInPly].data.dtype.names:
            features = plydata[nameInPly].data[name_feat].astype(np.float32).reshape(-1, 1)
            result['feat'] = torch.from_numpy(features).to(device)

    return result


def _load_from_txt(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, torch.Tensor]:
    """Read point cloud data from a text file.

    Args:
        filepath: Path to the text file

    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
    """
    # Load data - SLPCCD format has header lines that need to be skipped
    data = np.loadtxt(filepath, delimiter=' ', skiprows=2)

    # Extract XYZ coordinates
    positions = torch.from_numpy(data[:, 0:3].astype(np.float32)).to(device)
    result = {'pos': positions}

    # Extract features if available
    if data.shape[1] > 3:
        if data.shape[1] >= 7:
            # SLPCCD format: X Y Z Rf Gf Bf label - use label column as feature
            features = torch.from_numpy(data[:, 6:7].astype(np.float32)).to(device)
        else:
            # General format: use all remaining columns as features
            features = torch.from_numpy(data[:, 3:].astype(np.float32)).to(device)
        
        result['feat'] = features

    return result


def _load_from_las(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, torch.Tensor]:
    """Read point cloud data from a LAS/LAZ file.

    Args:
        filename: Path to the LAS/LAZ file

    Returns:
        Dictionary containing 'pos' and additional attributes
    """
    # Read the LAS/LAZ file
    las_file = laspy.read(filepath)

    # Extract XYZ coordinates
    points = np.vstack((las_file.x, las_file.y, las_file.z)).T.astype(np.float32)

    # Initialize result dictionary with position
    result = {'pos': torch.from_numpy(points).to(device)}

    # Extract RGB colors if available
    if all(field in las_file.point_format.dimension_names for field in ['red', 'green', 'blue']):
        # Normalize RGB values to [0, 1] range
        red = las_file.red / np.max(las_file.red)
        green = las_file.green / np.max(las_file.green)
        blue = las_file.blue / np.max(las_file.blue)
        rgb = np.vstack((red, green, blue)).T.astype(np.float32)
        result['rgb'] = torch.from_numpy(rgb).to(device)

    # Add all available attributes
    for field in las_file.point_format.dimension_names:
        if field not in ['x', 'y', 'z', 'red', 'green', 'blue']:  # Skip XYZ and RGB as they're already handled
            attr_value = getattr(las_file, field)
            if attr_value is not None:
                attr_value = np.array(attr_value, dtype=np.float32)
                attr_value = attr_value.reshape(-1, 1)
                result[field] = torch.from_numpy(attr_value).to(device)

    return result


def _load_from_pth(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, torch.Tensor]:
    """Load a point cloud from a PyTorch tensor file (.pth).
    
    Args:
        filepath: Path to the PyTorch tensor file (.pth)
        
    Returns:
        Dictionary with 'pos' containing XYZ coordinates and optional 'feat' for additional features
    """
    # Load the data - could be tensor, numpy array, or other format
    data = torch.load(filepath, map_location='cpu')
    
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    
    # Create result dictionary with device transfer
    result = {'pos': data[:, :3].to(device)}
    
    # Add features if available
    if data.shape[1] > 3:
        result['feat'] = data[:, 3:].to(device)
    
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
    device: Union[str, torch.device] = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Load a point cloud file and return in consistent dictionary format.

    Args:
        filepath: Path to point cloud file
        nameInPly: Name of vertex element in PLY file (optional)
        name_feat: Name of feature column (optional)
        device: Device to place tensors on ('cuda', 'cpu', or torch.device)

    Returns:
        Dictionary with at least 'pos' key containing XYZ coordinates as torch.float32
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
        pc_data = _load_from_ply(filepath, nameInPly=nameInPly, name_feat=name_feat, device=device)
    elif file_ext in ['.las', '.laz']:
        pc_data = _load_from_las(filepath, device=device)
    elif file_ext == '.off':
        pc_data = _load_from_off(filepath, device=device)
    elif file_ext == '.txt':
        pc_data = _load_from_txt(filepath, device=device)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # All loaders now return tensors on the correct device - use apply_tensor_op for normalization
    # Define normalization functions
    to_float32 = lambda x: x.float() if x.dtype != torch.float32 else x
    to_int64 = lambda x: x.long() if x.dtype != torch.int64 else x
    
    # Apply normalization to all tensors
    result = apply_tensor_op(to_float32, pc_data)
    
    # Validate that we have at least position data
    if 'pos' not in result:
        raise ValueError(f"Point cloud data must contain 'pos' field")
    
    # Validate position shape
    pos = result['pos']
    if pos.ndim != 2 or pos.shape[1] < 3:
        raise ValueError(f"Position data must have shape (N, D) where D >= 3, got {pos.shape}")
    
    # Handle segmentation files - convert labels to int64
    is_seg_file = '_seg' in os.path.basename(filepath).lower()
    if is_seg_file and 'feat' in result:
        result['feat'] = to_int64(result['feat'])
    
    # Validate result using input checks
    check_point_cloud(result)
    
    return result
