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
    # Normalize path to use forward slashes
    pathPC = os.path.normpath(pathPC).replace('\\', '/')
    
    if not os.path.isfile(pathPC):
        raise FileNotFoundError(f"Point cloud file not found: {pathPC}")
    
    # Check file extension 
    file_ext = os.path.splitext(pathPC)[1].lower()
    
    # Check if this is a segmentation file (_seg.txt) for SLPCCD dataset
    is_seg_file = '_seg' in os.path.basename(pathPC).lower()
    
    if file_ext == '.ply':
        pc_data = _read_from_ply(pathPC, nameInPly="params" if nameInPly is None else nameInPly, name_feat=name_feat)
    else:
        # For txt files and other formats, use specialized reader
        pc_data = _read_from_txt(pathPC)
        
        # For segmentation files, make sure the 4th column (labels) is correctly loaded
        # and converted to integer values for classification
        if is_seg_file:
            # Ensure 4th column (labels) is converted to int for classification
            pc_data[:, 3] = pc_data[:, 3].astype(np.int64)
    
    pc_data = torch.from_numpy(pc_data)
    return pc_data
