# Point Cloud IO Code Structure

## Code structure trees

`data/structures/three_d/point_cloud/io/load_point_cloud.py`

```text
load_point_cloud.py
├── import os
├── from typing import Any, Dict, Optional, Union
├── import laspy
├── import numpy as np
├── import open3d as o3d
├── import torch
├── from plyfile import PlyData
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── def load_point_cloud(filepath: str, meta_data: Optional[Dict[str, Dict[str, Any]]] = None, device: Union[str, torch.device] = 'cuda') -> PointCloud
│   ├── # Loads one point cloud file of any supported format as the cloud its own columns define, then applies the meta data over the halves that source leaves for the caller.
│   ├── def _validate_inputs [local]
│   │   └── assert the extension of filepath is one of the supported formats
│   ├── calls _validate_inputs()
│   ├── def _normalize_inputs [local]
│   │   ├── impls filepath = filepath with its separators rewritten to forward slashes
│   │   ├── assert filepath names an existing file  # output validation of the rewrite: the normalized path is the one that has to exist
│   │   └── return filepath
│   ├── calls _normalize_inputs(filepath=filepath)
│   ├── impls filepath = the value it returned
│   ├── calls _load_by_format(filepath=filepath, device=device)
│   ├── impls pc = the cloud it read, assembled as far as the source's own column names go and no further
│   ├── calls pc.apply_meta_data(meta_data=meta_data)
│   ├── assert pc carries an xyz field  # a raw cloud without coordinates is legal, a loaded one is not, so this is where a positional source that named no layout aborts
│   └── return pc
├── def _load_by_format(filepath: str, device: Union[str, torch.device]) -> PointCloud
│   ├── # Reads the file through the one reader that owns its extension.
│   ├── impls file_ext = the extension of filepath
│   ├── if file_ext == '.pth'
│   │   ├── calls _load_from_pth(filepath, device)
│   │   └── return  # the raw cloud it built
│   ├── if file_ext == '.ply'
│   │   ├── calls _load_from_ply(filepath, device)
│   │   └── return  # the raw cloud it built
│   ├── if file_ext == '.pcd'
│   │   ├── calls _load_from_pcd(filepath, device)
│   │   └── return  # the raw cloud it built
│   ├── if file_ext in ['.las', '.laz']
│   │   ├── calls _load_from_las(filepath, device)
│   │   └── return  # the raw cloud it built
│   ├── if file_ext == '.off'
│   │   ├── calls _load_from_off(filepath, device)
│   │   └── return  # the raw cloud it built
│   ├── if file_ext == '.txt'
│   │   ├── calls _load_from_txt(filepath, device)
│   │   └── return  # the raw cloud it built
│   └── assert 0, "Should not reach here."
├── def _load_from_pth(filepath: str, device: Union[str, torch.device]) -> PointCloud
│   ├── # Reads a .pth file holding one block the file names nothing about, each column becoming a field under its own index.
│   ├── calls torch.load(filepath, map_location='cpu')
│   ├── impls data = the loaded block
│   ├── assert data is a torch.Tensor or an np.ndarray
│   ├── assert data is two-dimensional  # a block of one axis has no columns to key by index, and splitting one raises an IndexError from inside the split rather than refusing the file at this door
│   ├── impls columns = each column of data as its own array, keyed by its index in the block as a string
│   ├── calls PointCloud(data=columns, device=device)
│   └── return  # the raw cloud it built, its coordinates unnamed until the caller's meta data names them
├── def _load_from_ply(filepath: str, device: Union[str, torch.device]) -> PointCloud
│   ├── # Reads a PLY's properties as fields, each in the dtype the file stores it in.
│   └── with open(filepath, "rb") as f
│       ├── calls PlyData.read(f)
│       ├── impls plydata = the parsed PLY
│       ├── assert plydata carries at least one element
│       ├── impls columns = every property of every element as its own contiguous array in the dtype the file stores it in, keyed by its property name and qualified as '<element>.<property>' when the file carries more than one  # impls-node-one-step:skip — names the key and the dtype
│       ├── calls PointCloud(data=columns, device=device)
│       └── return  # the raw cloud it built, a multi-element file qualifying every key so PLY's own coordinate names are absent from it
├── def _load_from_pcd(filepath: str, device: Union[str, torch.device]) -> PointCloud
│   ├── # Reads a PCD through Open3D's tensor IO, each attribute becoming a field whole under its own name.
│   ├── calls o3d.t.io.read_point_cloud(filepath)
│   ├── impls tensor_pcd = the read tensor point cloud
│   ├── impls columns = an empty dict
│   ├── for each attribute_name, ten in tensor_pcd.point.items()  # a TensorMap iterates its names alone, so the pairs come from items rather than from the map itself
│   │   └── impls columns[attribute_name] = ten handed straight to numpy, which is the one route carrying Open3D's unsigned widths and its bool  # an attribute is one named block, the way an in-memory variable is, so it is not split into columns of its own
│   ├── calls PointCloud(data=columns, device=device)
│   └── return  # the raw cloud it built
├── def _load_from_las(filepath: str, device: Union[str, torch.device]) -> PointCloud
│   ├── # Reads a LAS/LAZ file's dimensions as fields, each in the dtype laspy materializes it as, which makes a bit-packed dimension an ordinary uint8 field.
│   ├── calls laspy.read(filepath)
│   ├── impls las_file = the read LAS/LAZ file
│   ├── impls columns = an empty dict
│   ├── for each dimension_name in las_file.point_format.dimension_names
│   │   └── if dimension_name is not one of 'X', 'Y' and 'Z'
│   │       └── impls columns[dimension_name] = the attribute of las_file under that name, as a one-dimensional np.ndarray in the dtype laspy materialized it as
│   ├── impls columns['x'], columns['y'], columns['z'] = the three real-world coordinate arrays laspy scales the raw X, Y and Z dimensions into, each a one-dimensional float64 np.ndarray  # impls-node-one-step:skip — names the three coordinate columns
│   ├── calls PointCloud(data=columns, device=device)
│   └── return  # the raw cloud it built
├── def _load_from_off(filepath: str, device: Union[str, torch.device]) -> PointCloud
│   ├── # Reads the vertex block of an OFF file into float32 coordinate fields.
│   └── with open(filepath, 'r') as f
│       ├── impls header = the first line of f, stripped
│       ├── assert header starts with 'OFF'  # ModelNet40 writes the counts glued to the keyword, so OFF is the line's prefix rather than the whole of it
│       ├── impls counts_text = what follows the keyword on the header line when it carries anything, else the next line that is neither blank nor a comment  # impls-node-one-step:skip — names the one source of the counts across both header shapes
│       ├── impls n_vertices = the first of the three counts in counts_text
│       ├── impls vertices = an empty list
│       ├── for each of the n_vertices vertex lines that follow
│       │   ├── impls coords = the whitespace-separated floats of that line
│       │   └── impls vertices gains the leading three entries of coords
│       ├── impls positions = vertices as a float32 np.ndarray of shape [n_vertices, 3]  # float32 is the width this format is READ at, so the text lands there directly rather than being parsed wide and narrowed onto float32's grid afterwards, which every ordinary decimal would fail
│       ├── impls columns = the three columns of positions, keyed 'x', 'y' and 'z'  # a magnitude beyond float32 overflows in the parse, and validate_xyz_tensor is what aborts on it, this reader not repeating a check the construction it feeds already makes
│       ├── calls PointCloud(data=columns, device=device)
│       └── return  # the raw cloud it built
└── def _load_from_txt(filepath: str, device: Union[str, torch.device]) -> PointCloud
    ├── # Reads a whitespace-separated text point cloud whose leading two lines are a header, each column becoming a field under its own index and nothing divined from how many there are.
    ├── calls np.loadtxt(filepath, skiprows=2, dtype=np.float64, ndmin=2)  # no delimiter is named because the default splits on runs of whitespace, taking aligned columns and the single-space file alike, and ndmin holds the column axis open on the one-row file numpy would otherwise hand back flat
    ├── impls data = the parsed table  # ndmin holds the column axis open, since numpy drops it for a file of one row or one column and there would be no columns left to key by index
    ├── impls columns = each column of data as its own array, keyed by its index in the table as a string
    ├── calls PointCloud(data=columns, device=device)
    └── return  # the raw cloud it built, its coordinates unnamed until the caller's meta data names them
```

`data/structures/three_d/point_cloud/io/save_point_cloud.py`

```text
save_point_cloud.py
├── from typing import Any, Dict, Optional
├── import os
├── import numpy as np
├── from plyfile import PlyData, PlyElement
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from utils.dtypes import COLOR_RANGE, CONCEPTUAL_NAME, NUMPY_DTYPE, PLY_CHAR, cast_lossless
├── def save_point_cloud(pc: PointCloud, output_filepath: str, meta_data: Optional[Dict[str, Dict[str, Any]]] = None) -> None
│   ├── # Applies the meta data to the cloud and writes it through the writer that owns the output file's extension.
│   ├── def _validate_inputs [local]
│   │   ├── assert pc is a PointCloud
│   │   └── assert the extension of output_filepath is one of the supported formats  # the two doors of this module refuse an unsupported extension the same way, so a caller meets one behaviour rather than an assert on load and an exception on save
│   ├── calls _validate_inputs()
│   ├── def _normalize_inputs [local]
│   │   ├── impls output_filepath = output_filepath with its extension lowercased
│   │   └── return output_filepath
│   ├── calls _normalize_inputs(output_filepath=output_filepath)
│   ├── impls output_filepath = the value it returned
│   ├── calls pc.apply_meta_data(meta_data=meta_data)
│   ├── if the extension of output_filepath == '.ply'
│   │   └── calls _save_as_ply(pc, output_filepath)
│   └── return
└── def _save_as_ply(pc: PointCloud, output_filepath: str) -> None
    ├── # Writes a point cloud whose meta data is already applied, each field going to the columns and the ply dtype its entry names.
    ├── impls vertex_dtype = an empty list of (column name, ply dtype character) pairs
    ├── impls vertex_arrays = an empty dict
    ├── for each field_name in pc.field_names()
    │   ├── impls entry = pc.meta_data[field_name]
    │   ├── impls current_dtype = the 'dtype' of entry  # what the field means, which is the one thing an int32 tensor holding a uint16 colour cannot be asked
    │   ├── impls column_names = the 'layout' of entry
    │   ├── impls field_tensor = the field of pc under that name, detached, moved to cpu and given a column axis
    │   ├── if current_dtype sits in NUMPY_DTYPE
    │   │   └── impls field_data = field_tensor handed to numpy, still in the storage dtype torch held it in rather than the one it means  # a uint16 field arrives here as int32, which is what the cast below narrows back
    │   ├── else
    │   │   └── impls field_data = field_tensor cast in torch to torch.float32 and then handed to numpy  # bfloat16 has no numpy form at all, and f4 is the column PLY_CHAR sends it to anyway
    │   ├── assert column_names names exactly as many columns as field_data carries
    │   ├── assert no name in column_names is already a key of vertex_arrays  # two fields writing one ply column would silently overwrite each other
    │   ├── impls dtype_char = PLY_CHAR[current_dtype]  # an int64 field goes to i4 and a uint64 one to u4, and the per-column cast below is where the values decide whether that survives
    │   ├── if field_name == 'rgb'
    │   │   ├── assert current_dtype sits in COLOR_RANGE  # a colour whose dtype names no convention is refused rather than having one invented for it
    │   │   ├── impls source_low, source_high = COLOR_RANGE[CONCEPTUAL_NAME[the dtype of field_data]]
    │   │   ├── impls target_low, target_high = COLOR_RANGE[current_dtype]
    │   │   ├── impls converted = field_data mapped from the source bounds onto the target ones, rounded where the target names an integer convention
    │   │   ├── impls recovered = converted mapped back onto the source bounds
    │   │   ├── assert recovered equals field_data  # a file is read back, so a colour that rounded into the target grid would return as a different colour than the one saved
    │   │   └── impls field_data = converted
    │   └── for each column index i and column_name in column_names
    │       ├── calls cast_lossless(column i of field_data, np.dtype(dtype_char))
    │       ├── impls vertex_dtype gains the pair (column_name, dtype_char)
    │       └── impls vertex_arrays[column_name] = the column it cast
    ├── impls vertex_array = an empty structured array of pc.num_points rows and dtype vertex_dtype
    ├── for each column_name in vertex_arrays
    │   └── impls vertex_array[column_name] = vertex_arrays[column_name]
    ├── calls PlyElement.describe(vertex_array, 'vertex')
    ├── impls vertex_element = the described element
    ├── calls os.makedirs(the directory of output_filepath, exist_ok=True)
    └── calls PlyData([vertex_element]).write(output_filepath)
```
