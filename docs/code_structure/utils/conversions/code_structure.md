# Utils Conversions — code implementation structure

## Code implementation structure trees

`utils/conversions/depth_to_normals.py`

```text
depth_to_normals.py
├── import torch
├── import torch.nn.functional as F
└── def depth_to_normals(depth_map: torch.Tensor, camera_intrinsics: torch.Tensor, depth_ignore_value: float = float('inf'), normal_ignore_value: float = 0.0, return_mask: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ├── # Derives opencv camera-frame normals from a depth map: normal_ignore_value where depth is invalid or the normal NaN, else each pixel's L2-normalized cross product of its back-projected right and down neighbour vectors, zero on the last row and column.
    ├── assert depth_map is a torch.Tensor               # f"depth_map must be torch.Tensor, got {type(depth_map)}"
    ├── assert depth_map is 2-D                          # f"depth_map must be 2D tensor [H, W], got shape {depth_map.shape}"
    ├── assert camera_intrinsics is a torch.Tensor       # f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    ├── assert camera_intrinsics is 3 x 3                # f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    ├── assert depth_ignore_value is an int or a float   # f"depth_ignore_value must be numeric, got {type(depth_ignore_value)}"
    ├── assert normal_ignore_value is an int or a float  # f"normal_ignore_value must be numeric, got {type(normal_ignore_value)}"
    ├── impls H, W = the shape of depth_map
    ├── impls device = the device of depth_map
    ├── impls fx = the (0, 0) entry of camera_intrinsics
    ├── impls fy = the (1, 1) entry of camera_intrinsics
    ├── impls cx = the (0, 2) entry of camera_intrinsics
    ├── impls cy = the (1, 2) entry of camera_intrinsics
    ├── assert fx > 0  # f"fx must be positive, got {fx}"
    ├── assert fy > 0  # f"fy must be positive, got {fy}"
    ├── impls u, v = the xy-indexed meshgrid of the column indices 0 to W - 1 by the row indices 0 to H - 1, on device
    ├── impls u = u as float
    ├── impls v = v as float
    ├── if depth_ignore_value is infinite
    │   └── impls valid_mask = the pixels where depth_map is finite
    ├── else
    │   └── impls valid_mask = the pixels where depth_map differs from depth_ignore_value
    ├── impls Z = depth_map
    ├── impls X = (u - cx) * Z / fx
    ├── impls Y = (v - cy) * Z / fy
    ├── impls u_right = u + 1 clamped to at most W - 1
    ├── impls Z_right = a zeros tensor like depth_map
    ├── impls assign depth_map's columns 1 to W - 1 into Z_right's columns 0 to W - 2
    ├── impls assign depth_map's last column into Z_right's last column
    ├── impls X_right = (u_right - cx) * Z_right / fx
    ├── impls Y_right = (v - cy) * Z_right / fy
    ├── impls v_down = v + 1 clamped to at most H - 1
    ├── impls Z_down = a zeros tensor like depth_map
    ├── impls assign depth_map's rows 1 to H - 1 into Z_down's rows 0 to H - 2
    ├── impls assign depth_map's last row into Z_down's last row
    ├── impls X_down = (u - cx) * Z_down / fx
    ├── impls Y_down = (v_down - cy) * Z_down / fy
    ├── impls vec_right = the dim-0 stack of X_right - X, Y_right - Y, Z_right - Z
    ├── impls vec_down = the dim-0 stack of X_down - X, Y_down - Y, Z_down - Z
    ├── impls normals = the dim-0 cross product of vec_right with vec_down
    ├── impls normals = F.normalize of normals over dim 0, dividing by their L2 norm clamped to at least 1e-12
    ├── impls invalid_mask = the complement of valid_mask
    ├── impls nan_mask = the pixels where any normals channel is NaN
    ├── impls final_invalid_mask = invalid_mask or nan_mask, pixel by pixel
    ├── impls assign normal_ignore_value into every channel of normals at final_invalid_mask
    ├── if return_mask
    │   └── return normals, the complement of final_invalid_mask
    └── else
        └── return normals
```
