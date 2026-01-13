from typing import Dict, Any
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def _save_as_ply(pc: PointCloud, output_filepath: str) -> None:
    """Save point cloud data to PLY file using plyfile library.

    Args:
        pc: Point cloud data with xyz and other fields
        output_filepath: Output file path (must end with .ply)
    """
    assert output_filepath.endswith(
        '.ply'
    ), f"Output file must be .ply format, got: {output_filepath}"

    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    field_mapping: Dict[str, Any] = {
        name: getattr(pc, name) for name in pc.field_names()
    }

    # Get positions and convert to numpy if needed
    positions = field_mapping['xyz']
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    assert (
        len(positions.shape) == 2 and positions.shape[1] == 3
    ), f"Expected positions shape (N, 3), got: {positions.shape}"

    num_points = positions.shape[0]

    # Build vertex dtype and data dictionary dynamically
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_arrays = {
        'x': positions[:, 0].astype(np.float32),
        'y': positions[:, 1].astype(np.float32),
        'z': positions[:, 2].astype(np.float32),
    }

    # Process all other fields dynamically
    for field_name, field_data in field_mapping.items():
        if field_name in ('xyz', 'pos') or field_data is None:
            continue

        # Convert to numpy if needed
        if isinstance(field_data, torch.Tensor):
            field_data = field_data.detach().cpu().numpy()

        # Handle special color field mappings
        if field_name in ['colors', 'rgb'] and field_data.shape[1] == 3:
            # Map to standard PLY color names
            color_data = field_data
            if color_data.max() <= 1.0:
                color_data = (color_data * 255).astype(np.uint8)
            else:
                color_data = color_data.astype(np.uint8)

            vertex_dtype.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            vertex_arrays['red'] = color_data[:, 0]
            vertex_arrays['green'] = color_data[:, 1]
            vertex_arrays['blue'] = color_data[:, 2]

        elif field_name == 'normals' and field_data.shape[1] == 3:
            # Map to standard PLY normal names
            normal_data = field_data.astype(np.float32)
            vertex_dtype.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
            vertex_arrays['nx'] = normal_data[:, 0]
            vertex_arrays['ny'] = normal_data[:, 1]
            vertex_arrays['nz'] = normal_data[:, 2]

        else:
            # Handle arbitrary fields dynamically
            if len(field_data.shape) == 1:
                # Single-column field
                if field_data.dtype.kind in ['i', 'u']:  # Integer types
                    dtype_char = 'i4'  # Always use i4 for integers
                else:  # Float types
                    dtype_char = 'f4' if field_data.dtype.itemsize <= 4 else 'f8'

                vertex_dtype.append((field_name, dtype_char))
                vertex_arrays[field_name] = field_data.astype(
                    dtype_char[0] + str(int(dtype_char[1]))
                )

            elif len(field_data.shape) == 2:
                # Multi-column field - create separate entries for each column
                for i in range(field_data.shape[1]):
                    col_name = (
                        f"{field_name}_{i}" if field_data.shape[1] > 1 else field_name
                    )

                    if field_data.dtype.kind in ['i', 'u']:  # Integer types
                        dtype_char = 'i4'  # Always use i4 for integers
                    else:  # Float types
                        dtype_char = 'f4' if field_data.dtype.itemsize <= 4 else 'f8'

                    vertex_dtype.append((col_name, dtype_char))
                    vertex_arrays[col_name] = field_data[:, i].astype(
                        dtype_char[0] + str(int(dtype_char[1]))
                    )

    # Create structured numpy array efficiently
    vertex_array = np.empty(num_points, dtype=vertex_dtype)
    for field_name in vertex_arrays:
        vertex_array[field_name] = vertex_arrays[field_name]

    # Create PLY element
    vertex_element = PlyElement.describe(vertex_array, 'vertex')

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Write PLY file
    PlyData([vertex_element]).write(output_filepath)


def save_point_cloud(pc: PointCloud, output_filepath: str) -> None:
    """Save point cloud data to file.

    Args:
        pc: Point cloud data with xyz and other fields
        output_filepath: Output file path (supports .ply format)
    """
    file_ext = os.path.splitext(output_filepath)[1].lower()

    if file_ext == '.ply':
        _save_as_ply(pc, output_filepath)
    else:
        raise ValueError(
            f"Unsupported output format: {file_ext}. Currently only .ply is supported."
        )

    # Count points for logging
    num_points = pc.num_points

    print(f"   💾 Saved {num_points} points to: {output_filepath}")
