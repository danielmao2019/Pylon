import os
from typing import Any, Dict, Optional

import numpy as np
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from utils.dtypes import NUMPY_DTYPE, PLY_CHAR, cast_lossless


def save_point_cloud(
    pc: PointCloud,
    output_filepath: str,
    meta_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Applies the meta data to the cloud and writes it through the writer that owns the output file's extension.

    Args:
        pc: The point cloud to write, as a PointCloud whose fields are still on the record it carries.
        output_filepath: The path of the file to write, as a str carrying the '.ply' extension.
        meta_data: The override, one entry per field keyed by field name, each entry stating a 'dtype' naming a conceptual dtype, a 'layout' naming the output columns as a tuple of str, or both, or None to write the cloud at the record it carries.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(
            pc, PointCloud
        ), f"a point cloud is written from a PointCloud: type(pc)={type(pc)}"
        # the two doors of this module refuse an unsupported extension the same way, so a caller meets one behaviour rather than an assert on load and an exception on save
        assert os.path.splitext(output_filepath)[1].lower() in (
            '.ply',
        ), f"a file is written by the writer that owns its extension, and no writer owns this one: output_filepath={output_filepath}, extension={os.path.splitext(output_filepath)[1]}"

    _validate_inputs()

    def _normalize_inputs(output_filepath: str) -> str:
        output_filepath = (
            os.path.splitext(output_filepath)[0]
            + os.path.splitext(output_filepath)[1].lower()
        )
        return output_filepath

    output_filepath = _normalize_inputs(output_filepath=output_filepath)

    pc.apply_meta_data(meta_data=meta_data)
    if os.path.splitext(output_filepath)[1] == '.ply':
        _save_as_ply(pc, output_filepath)

    return


def _save_as_ply(pc: PointCloud, output_filepath: str) -> None:
    """Writes a point cloud whose meta data is already applied, each field going to the columns and the ply dtype its own entry names, with every value already on the convention and the width that entry states.

    Args:
        pc: The point cloud to write, as a PointCloud whose meta data is already applied, every field carrying the values of the convention and the width its own entry names.
        output_filepath: The path of the .ply file to write, as a str.

    Returns:
        None.
    """
    vertex_dtype = []
    vertex_arrays = {}
    for field_name in pc.field_names():
        entry = pc.meta_data[field_name]
        # what the field means, which is the one thing an int32 tensor holding a uint16 colour cannot be asked
        current_dtype = entry['dtype']
        column_names = entry['layout']
        field_tensor = getattr(pc, field_name).detach().cpu().reshape(pc.num_points, -1)
        if current_dtype in NUMPY_DTYPE:
            # a uint16 field arrives here as int32, which is what the cast below narrows back
            field_data = field_tensor.numpy()
        else:
            # bfloat16 has no numpy form at all, and f4 is the column PLY_CHAR sends it to anyway
            field_data = field_tensor.float().numpy()
        assert len(column_names) == field_data.shape[1], (
            "the reverse mapping writes one output column per name, so a count that disagrees leaves the writer with no name for a column: "
            f"field_name={field_name}, layout={column_names}, field_data.shape={field_data.shape}"
        )
        # two fields writing one ply column would silently overwrite each other
        assert all(
            column_name not in vertex_arrays for column_name in column_names
        ), f"one ply column is written by one field: field_name={field_name}, layout={column_names}, columns already written={tuple(vertex_arrays.keys())}"
        # an int64 field goes to i4 and a uint64 one to u4, and the per-column cast below is where the values decide whether that survives
        dtype_char = PLY_CHAR[current_dtype]
        for i, column_name in enumerate(column_names):
            vertex_dtype.append((column_name, dtype_char))
            vertex_arrays[column_name] = cast_lossless(
                values=field_data[:, i], dtype=np.dtype(dtype_char)
            )

    vertex_array = np.empty(pc.num_points, dtype=vertex_dtype)
    for column_name in vertex_arrays:
        vertex_array[column_name] = vertex_arrays[column_name]
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    PlyData([vertex_element]).write(output_filepath)
