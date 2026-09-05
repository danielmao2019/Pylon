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
│   │   ├── assert the normalized form of filepath names an existing file
│   │   └── assert the lowercased extension of filepath is one of the supported formats
│   ├── calls _validate_inputs()
│   ├── def _normalize_inputs [local]
│   │   ├── impls filepath = filepath normalized with its separators rewritten to forward slashes
│   │   └── return filepath
│   ├── calls _normalize_inputs(filepath=filepath)
│   ├── impls filepath = the returned value from _normalize_inputs
│   ├── impls file_ext = the lowercased extension of filepath
│   ├── def _load_by_format [local]
│   │   ├── # Reads the file through the one reader that owns its extension.
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
│   ├── def numpy_to_torch_on_device(key: str, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor [local]
│   │   ├── # Places one loaded field on the requested device, casting to the requested dtype only the positions.
│   │   ├── if isinstance(x, np.ndarray)
│   │   │   ├── if x.dtype == np.uint16
│   │   │   │   └── impls x = x cast to np.int32  # torch carries no uint16, and int32 inflates it least
│   │   │   ├── impls tensor = x wrapped as a torch tensor
│   │   │   ├── impls tensor = tensor moved to device
│   │   │   ├── if key == 'xyz'
│   │   │   │   └── impls tensor = tensor cast to dtype
│   │   │   └── return tensor
│   │   ├── elif isinstance(x, torch.Tensor)
│   │   │   ├── impls tensor = x moved to device
│   │   │   ├── if key == 'xyz'
│   │   │   │   └── impls tensor = tensor cast to dtype
│   │   │   └── return tensor
│   │   └── else
│   │       └── return x
│   ├── for each key, value in pc_data
│   │   └── calls numpy_to_torch_on_device(key, value)
│   ├── impls result = the placed fields under the keys pc_data carried them under
│   ├── assert result carries 'xyz'
│   ├── impls is_seg_file = whether '_seg' occurs in the lowercased basename of filepath
│   ├── if is_seg_file and result carries 'feat'
│   │   └── impls result['feat'] = result['feat'] cast to int64  # a segmentation file's feature column is a label column
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
