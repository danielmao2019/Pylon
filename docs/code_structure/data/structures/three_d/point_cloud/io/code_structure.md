# Point Cloud IO Code Structure

## Code structure trees

`data/structures/three_d/point_cloud/io/load_point_cloud.py`

```text
load_point_cloud.py
├── import os
├── from typing import Dict, Optional, Union
├── import laspy
├── import numpy as np
├── import open3d as o3d
├── import torch
├── import torch.utils.dlpack as torch_dlpack
├── from plyfile import PlyData
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── def load_point_cloud(filepath: str, nameInPly: Optional[str] = None, name_feat: Optional[str] = None, device: Union[str, torch.device] = 'cuda', dtype: torch.dtype = torch.float32) -> PointCloud
│   ├── # Loads one point cloud file of any supported format and hands it back as a PointCloud placed on the requested device.
│   ├── def _validate_inputs [local]
│   │   └── assert the extension of filepath is one of the supported formats
│   ├── calls _validate_inputs()
│   ├── def _normalize_inputs [local]
│   │   ├── impls filepath = filepath normalized with its separators rewritten to forward slashes
│   │   ├── assert filepath names an existing file
│   │   └── return filepath
│   ├── calls _normalize_inputs(filepath=filepath)
│   ├── impls filepath = the returned value from _normalize_inputs
│   ├── def _load_by_format [local]
│   │   ├── # Reads the file through the one reader that owns its extension.
│   │   ├── impls file_ext = the extension of filepath
│   │   ├── if file_ext == '.pth'
│   │   │   ├── calls _load_from_pth(filepath, device=device)
│   │   │   └── return the fields it read
│   │   ├── if file_ext == '.ply'
│   │   │   ├── calls _load_from_ply(filepath, nameInPly=nameInPly, name_feat=name_feat)
│   │   │   └── return the fields it read
│   │   ├── if file_ext == '.pcd'
│   │   │   ├── calls _load_from_pcd(filepath)
│   │   │   └── return the fields it read
│   │   ├── if file_ext in ['.las', '.laz']
│   │   │   ├── calls _load_from_las(filepath)
│   │   │   └── return the fields it read
│   │   ├── if file_ext == '.off'
│   │   │   ├── calls _load_from_off(filepath, device=device)
│   │   │   └── return the fields it read
│   │   ├── if file_ext == '.txt'
│   │   │   ├── calls _load_from_txt(filepath)
│   │   │   └── return the fields it read
│   │   └── assert 0, "Should not reach here."
│   ├── calls _load_by_format()
│   ├── impls pc_data = the fields the matched reader returned
│   ├── def _normalize_field(key: str, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor [local]
│   │   ├── # Places one loaded field on the requested device and casts it to the dtype its key calls for.
│   │   ├── if isinstance(x, np.ndarray)
│   │   │   ├── if x.dtype == np.uint16
│   │   │   │   └── impls x = x cast to np.int32  # torch carries no uint16, and int32 inflates it least
│   │   │   └── impls x = x wrapped as a torch tensor
│   │   ├── assert x is a torch.Tensor
│   │   ├── impls tensor = x moved to device
│   │   ├── if key == 'xyz'
│   │   │   ├── assert tensor is a floating point tensor
│   │   │   └── impls tensor = tensor cast to dtype
│   │   ├── impls is_seg_file = whether '_seg' occurs in the basename of filepath
│   │   ├── if key == 'indices'
│   │   │   ├── assert tensor is an integer tensor
│   │   │   └── impls tensor = tensor cast to int64  # PointCloud consumes indices as an index tensor
│   │   ├── if key == 'feat' and is_seg_file
│   │   │   ├── assert tensor is a floating point or integer tensor
│   │   │   └── impls tensor = tensor cast to int64  # a segmentation file's feature column is a label column
│   │   └── return tensor
│   ├── impls result = an empty dict
│   ├── for each key, value in pc_data
│   │   ├── calls _normalize_field(key, value)
│   │   └── impls result[key] = the normalized field it returned
│   ├── calls PointCloud(data=result)
│   └── return  # the PointCloud wrapping result
├── def _load_from_pth(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, Union[torch.Tensor, np.ndarray]]
│   ├── # Reads a .pth file holding one [N, >= 3] block whose leading three columns are the coordinates.
│   ├── calls torch.load(filepath, map_location='cpu')
│   ├── impls data = the loaded block
│   ├── assert data is a torch.Tensor or an np.ndarray
│   ├── impls result = {'xyz': the first three columns of data}
│   ├── if data.shape[1] > 3
│   │   └── impls result['feat'] = the columns of data past the third
│   └── return result
├── def _load_from_ply(filepath: str, nameInPly: Optional[str] = None, name_feat: Optional[str] = None) -> Dict[str, np.ndarray]
│   ├── # Reads one PLY element into float64 coordinates plus every other field that element carries.
│   ├── with open(filepath, "rb") as f
│   │   ├── calls PlyData.read(f)
│   │   ├── impls plydata = the parsed PLY
│   │   ├── if nameInPly is None
│   │   │   ├── assert plydata carries exactly one element
│   │   │   └── impls nameInPly = the name of that one element
│   │   ├── impls num_verts = the vertex count of plydata[nameInPly]
│   │   ├── impls available_fields = the field names of plydata[nameInPly]
│   │   ├── impls positions = a zeros array of shape [num_verts, 3] and dtype np.float64  # impls-node-one-step:skip — names the shape and the dtype
│   │   ├── impls positions column 0 = the x field cast to np.float64
│   │   ├── impls positions column 1 = the y field cast to np.float64
│   │   ├── impls positions column 2 = the z field cast to np.float64
│   │   ├── impls result = {'xyz': positions}
│   │   ├── if available_fields carries red, green and blue
│   │   │   ├── impls rgb = the red, green and blue fields column-stacked, in the dtype and range they are stored in  # impls-node-one-step:skip — names the fields
│   │   │   └── impls result['rgb'] = rgb made contiguous
│   │   ├── for each field_name in available_fields
│   │   │   └── if field_name is none of x, y, z, red, green and blue
│   │   │       ├── impls field_array = the field made contiguous, in the shape and dtype it is stored in  # impls-node-one-step:skip — names shape and dtype
│   │   │       ├── if field_array is two-dimensional with a trailing axis of one
│   │   │       │   └── impls field_array = field_array with that trailing axis squeezed out
│   │   │       └── impls result[field_name] = field_array
│   │   └── if name_feat is given, sits in available_fields and is absent from result
│   │       ├── impls features = the name_feat field cast to np.float64
│   │       ├── impls features = features reshaped to [N, 1]
│   │       └── impls result['feat'] = features
│   └── return result
├── def _load_from_pcd(filepath: str) -> Dict[str, torch.Tensor]
│   ├── # Reads a PCD through Open3D's tensor IO, handing every attribute to torch by DLPack in the shape and dtype it was stored in.
│   ├── calls o3d.t.io.read_point_cloud(filepath)
│   ├── impls tensor_pcd = the read tensor point cloud
│   ├── assert tensor_pcd.point carries 'positions'
│   ├── impls pos_t = tensor_pcd.point['positions'] handed to torch through torch_dlpack.from_dlpack
│   ├── impls result = {'xyz': pos_t}
│   ├── if tensor_pcd.point carries 'colors'
│   │   └── impls result['rgb'] = tensor_pcd.point['colors'] handed to torch through torch_dlpack.from_dlpack
│   ├── for each field_name, ten in tensor_pcd.point
│   │   ├── if field_name is 'positions' or 'colors'
│   │   │   └── continue
│   │   └── impls result[field_name] = ten handed to torch through torch_dlpack.from_dlpack
│   └── return result
├── def _load_from_las(filepath: str) -> Dict[str, np.ndarray]
│   ├── # Reads a LAS/LAZ file into float64 coordinates plus every dimension its point format declares.
│   ├── calls laspy.read(filepath)
│   ├── impls las_file = the read LAS/LAZ file
│   ├── impls points = the x, y and z dimensions cast to np.float64  # impls-node-one-step:skip — names the three dimensions
│   ├── impls points = points stacked into [N, 3]
│   ├── impls result = {'xyz': points}
│   ├── if las_file.point_format.dimension_names carries red, green and blue
│   │   ├── impls rgb = the red, green and blue dimensions stacked into [N, 3], in the dtype and range they are stored in  # impls-node-one-step:skip — names the dimensions
│   │   └── impls result['rgb'] = rgb
│   ├── for each field in las_file.point_format.dimension_names
│   │   └── if field is none of x, y, z, red, green and blue
│   │       ├── impls attr_value = the attribute of las_file under that name
│   │       └── if attr_value is not None
│   │           ├── impls attr_value = attr_value as an np.ndarray, in the shape and dtype it is stored in  # impls-node-one-step:skip — names shape and dtype
│   │           ├── if attr_value is two-dimensional with a trailing axis of one
│   │           │   └── impls attr_value = attr_value with that trailing axis squeezed out
│   │           └── impls result[field] = attr_value
│   └── return result
├── def _load_from_off(filepath: str, device: Union[str, torch.device] = 'cuda') -> Dict[str, torch.Tensor]
│   ├── # Reads the vertex block of an OFF file, keeping the leading three columns of each vertex line.
│   └── with open(filepath, 'r') as f
│       ├── impls header = the first line of f, stripped
│       ├── assert header == 'OFF'
│       ├── impls n_vertices = the first of the three counts on the next line
│       ├── impls vertices = an empty list
│       ├── for each of the n_vertices vertex lines that follow
│       │   ├── impls coords = the whitespace-separated floats of that line
│       │   └── impls vertices gains the leading three entries of coords
│       ├── calls torch.tensor(vertices, dtype=torch.float32, device=device)
│       ├── impls positions = the built tensor
│       ├── impls result = {'xyz': positions}
│       └── return result
└── def _load_from_txt(filepath: str) -> Dict[str, np.ndarray]
    ├── # Reads a space-separated text point cloud whose leading two lines are a header.
    ├── calls np.loadtxt(filepath, delimiter=' ', skiprows=2, dtype=np.float64)
    ├── impls data = the parsed table
    ├── impls positions = the first three columns of data
    ├── impls result = {'xyz': positions}
    ├── if data.shape[1] > 3
    │   ├── if data.shape[1] >= 7
    │   │   └── impls features = column six of data, kept as an [N, 1] block  # the label column of the SLPCCD X Y Z Rf Gf Bf label layout
    │   ├── else
    │   │   └── impls features = the columns of data past the third
    │   └── impls result['feat'] = features
    └── return result
```

`data/structures/three_d/point_cloud/io/save_point_cloud.py`

```text
save_point_cloud.py
├── from typing import Dict, Any
├── import os
├── import numpy as np
├── import torch
├── from plyfile import PlyData, PlyElement
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── def save_point_cloud(pc: PointCloud, output_filepath: str) -> None
│   ├── # Writes a point cloud through the writer that owns the output file's extension.
│   ├── impls file_ext = the lowercased extension of output_filepath
│   ├── if file_ext == '.ply'
│   │   └── calls _save_as_ply(pc, output_filepath)
│   ├── else
│   │   └── raise ValueError  # the extension matches no supported format
│   ├── impls num_points = pc.num_points
│   └── impls prints a line naming num_points and output_filepath
└── def _save_as_ply(pc: PointCloud, output_filepath: str) -> None
    ├── # Writes a point cloud to a PLY file, mapping each field onto the vertex columns the format names.
    ├── assert output_filepath ends with '.ply'
    ├── assert pc is a PointCloud
    ├── impls field_mapping: Dict[str, Any] = every field of pc under its own name
    ├── impls positions = field_mapping['xyz']
    ├── if positions is a torch.Tensor
    │   └── impls positions = positions detached, moved to cpu and handed to numpy
    ├── assert positions is two-dimensional with a trailing axis of three
    ├── impls num_points = the row count of positions
    ├── impls vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    ├── impls vertex_arrays = the x, y and z position columns cast to np.float32, keyed by those names
    ├── for each field_name, field_data in field_mapping
    │   ├── if field_name is 'xyz' or 'pos', or field_data is None
    │   │   └── continue
    │   ├── if field_data is a torch.Tensor
    │   │   └── impls field_data = field_data detached, moved to cpu and handed to numpy
    │   ├── if field_name is 'colors' or 'rgb', and field_data has three columns
    │   │   ├── impls color_data = field_data
    │   │   ├── if color_data.max() <= 1.0
    │   │   │   └── impls color_data = color_data scaled by 255 and cast to np.uint8
    │   │   ├── else
    │   │   │   └── impls color_data = color_data cast to np.uint8
    │   │   ├── impls vertex_dtype gains ('red', 'u1'), ('green', 'u1') and ('blue', 'u1')
    │   │   ├── impls vertex_arrays['red'] = column 0 of color_data
    │   │   ├── impls vertex_arrays['green'] = column 1 of color_data
    │   │   └── impls vertex_arrays['blue'] = column 2 of color_data
    │   ├── elif field_name is 'normals', and field_data has three columns
    │   │   ├── impls normal_data = field_data cast to np.float32
    │   │   ├── impls vertex_dtype gains ('nx', 'f4'), ('ny', 'f4') and ('nz', 'f4')
    │   │   ├── impls vertex_arrays['nx'] = column 0 of normal_data
    │   │   ├── impls vertex_arrays['ny'] = column 1 of normal_data
    │   │   └── impls vertex_arrays['nz'] = column 2 of normal_data
    │   └── else
    │       ├── if field_data is one-dimensional
    │       │   ├── if field_data.dtype.kind is 'i' or 'u'
    │       │   │   ├── assert every value of field_data fits in int32
    │       │   │   └── impls dtype_char = 'i4'  # PLY carries no 64-bit integer
    │       │   ├── else
    │       │   │   └── impls dtype_char = 'f4' when field_data.dtype.itemsize is at most 4, else 'f8'
    │       │   ├── impls vertex_dtype gains (field_name, dtype_char)
    │       │   └── impls vertex_arrays[field_name] = field_data cast to dtype_char  # the dtype string is rebuilt from its own characters
    │       └── elif field_data is two-dimensional
    │           └── for each column i of field_data
    │               ├── impls col_name = field_name suffixed with i when field_data has more than one column, else field_name
    │               ├── if field_data.dtype.kind is 'i' or 'u'
    │               │   ├── assert every value of that column fits in int32
    │               │   └── impls dtype_char = 'i4'  # PLY carries no 64-bit integer
    │               ├── else
    │               │   └── impls dtype_char = 'f4' when field_data.dtype.itemsize is at most 4, else 'f8'
    │               ├── impls vertex_dtype gains (col_name, dtype_char)
    │               └── impls vertex_arrays[col_name] = column i of field_data cast to dtype_char  # the dtype string is rebuilt from its own characters
    ├── impls vertex_array = an empty structured array of num_points rows and dtype vertex_dtype
    ├── for each field_name in vertex_arrays
    │   └── impls vertex_array[field_name] = vertex_arrays[field_name]
    ├── calls PlyElement.describe(vertex_array, 'vertex')
    ├── impls vertex_element = the described element
    ├── calls os.makedirs(the directory of output_filepath, exist_ok=True)
    └── calls PlyData([vertex_element]).write(output_filepath)
```
