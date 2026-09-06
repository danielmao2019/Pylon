# Point Cloud Data Structure Code Structure

## Code structure trees

`data/structures/three_d/point_cloud/point_cloud.py`

```text
point_cloud.py
├── from typing import Dict, Optional, Tuple
├── import torch
└── class PointCloud
    ├── # One point cloud: an xyz coordinate field plus arbitrary named per-point fields, every one a torch tensor of the same length on one device.
    ├── def __init__(self, xyz: Optional[torch.Tensor] = None, data: Optional[Dict[str, torch.Tensor]] = None) -> None
    │   ├── # Builds a point cloud from a coordinate tensor, from a field dict carrying its own coordinates, or from both.
    │   ├── assert xyz is None or xyz is a torch.Tensor
    │   ├── assert data is None or data is a dict
    │   ├── if data is not None
    │   │   └── assert every key of data is a str
    │   ├── if xyz is None
    │   │   ├── assert data is not None
    │   │   ├── assert data carries 'xyz'
    │   │   └── impls xyz = data['xyz']
    │   ├── else
    │   │   └── assert data is None or data carries no 'xyz'
    │   ├── assert xyz is a torch.Tensor
    │   ├── impls _fields = an empty dict
    │   ├── impls _length = the row count of xyz
    │   ├── impls _device = the device of xyz
    │   ├── calls self._validate_field(name='xyz', value=xyz)
    │   ├── impls _xyz = xyz
    │   └── if data is not None
    │       └── for each key, value in data
    │           ├── if key == 'xyz'
    │           │   └── continue
    │           ├── calls self._assert_field_name_valid(name=key)
    │           ├── calls self._validate_field(name=key, value=value)
    │           └── impls _fields[key] = value
    ├── @property def xyz(self) -> torch.Tensor
    │   ├── # Hands back the coordinate field.
    │   └── return self._xyz
    ├── @property def device(self) -> torch.device
    │   ├── # Hands back the one device every field of this point cloud sits on.
    │   └── return self._device
    ├── @property def num_points(self) -> int
    │   ├── # Hands back the number of points every field carries.
    │   └── return self._length
    ├── def __len__(self) -> int
    │   ├── # Serves the point count to len(), so a point cloud measures as its number of points.
    │   └── return self._length
    ├── def field_names(self) -> Tuple[str, ...]
    │   ├── # Hands back every field name this point cloud carries, coordinates first.
    │   ├── impls names = 'xyz' followed by the keys of self._fields
    │   └── return names
    ├── def __getattr__(self, name: str) -> torch.Tensor
    │   ├── # Serves a named field as an attribute, for a name ordinary attribute lookup did not find.
    │   ├── assert each of '_xyz', '_fields', '_length' and '_device' sits in self.__dict__
    │   ├── if name sits in self._fields
    │   │   └── return self._fields[name]
    │   └── raise AttributeError  # the name is no field this point cloud carries
    ├── def __setattr__(self, name: str, value: torch.Tensor) -> None
    │   ├── # Routes an assignment to the private slot for an underscore name, and to a validated field otherwise.
    │   ├── if name starts with '_'
    │   │   ├── impls the value goes to the slot through the base class attribute setter
    │   │   └── return
    │   ├── calls self._assert_field_name_valid(name=name)
    │   ├── calls self._validate_field(name=name, value=value)
    │   ├── if name == 'xyz'
    │   │   └── impls _xyz = value
    │   └── else
    │       └── impls _fields[name] = value
    ├── def __getstate__(self) -> dict
    │   ├── # Hands the four private slots to pickle, so a point cloud survives a round trip across a process boundary.
    │   ├── impls state = the four private slots _xyz, _fields, _length and _device keyed by their own names  # impls-node-one-step:skip
    │   └── return state
    ├── def __setstate__(self, state: dict) -> None
    │   ├── # Restores the four private slots from a pickled state dict.
    │   ├── assert state is a dict
    │   ├── assert state carries '_xyz'
    │   ├── assert state carries '_fields'
    │   ├── assert state carries '_length'
    │   ├── assert state carries '_device'
    │   ├── impls _xyz = state['_xyz']
    │   ├── impls _fields = state['_fields']
    │   ├── impls _length = state['_length']
    │   └── impls _device = state['_device']
    ├── def _validate_field(self, name: str, value: torch.Tensor) -> None
    │   ├── # Checks one field's tensor-ness, rank, length and device, then the extra rules the names xyz, rgb and indices carry.
    │   ├── assert value is a torch.Tensor
    │   ├── assert value is at least one-dimensional
    │   ├── assert value carries at least one point
    │   ├── assert the row count of value matches self._length
    │   ├── assert the device of value matches self._device
    │   ├── if name == 'xyz'
    │   │   └── calls self.validate_xyz_tensor(value)
    │   ├── elif name == 'rgb'
    │   │   └── calls self.validate_rgb_tensor(value)
    │   └── elif name == 'indices'
    │       └── assert value.dtype is torch.int64
    ├── @staticmethod def validate_xyz_tensor(xyz: torch.Tensor) -> None
    │   ├── # Checks coordinates are an [N, 3] floating point tensor free of NaN and Inf.
    │   ├── assert xyz is a torch.Tensor
    │   ├── assert xyz is two-dimensional
    │   ├── assert xyz has three columns
    │   ├── assert xyz is a floating point tensor
    │   ├── assert xyz carries no NaN
    │   └── assert xyz carries no Inf
    ├── @staticmethod def validate_rgb_tensor(rgb: torch.Tensor) -> None
    │   ├── # Checks colors are an [N, 3] tensor free of NaN and Inf.
    │   ├── assert rgb is a torch.Tensor
    │   ├── assert rgb is two-dimensional
    │   ├── assert rgb has three columns
    │   ├── assert rgb carries no NaN
    │   └── assert rgb carries no Inf
    └── def _assert_field_name_valid(self, name: str) -> None
        ├── # Checks a field name is a str, is not underscore-prefixed, and collides with none of the reserved attribute names.
        ├── assert name is a str
        ├── assert name does not start with '_'
        └── assert name is none of 'device', 'num_points' and 'field_names'
```

`data/structures/three_d/point_cloud/select.py`

```text
select.py
├── from typing import Dict, List, Union
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class Select
    ├── # Indexes a point cloud down to the points a fixed index list or index tensor names.
    ├── def __init__(self, indices: Union[torch.Tensor, List[int]]) -> None
    │   ├── # Holds the indices this selection will take, in the list or tensor form it was given.
    │   └── impls self.indices = indices
    ├── def __call__(self, pc: PointCloud) -> PointCloud
    │   ├── # Builds a new point cloud carrying every field of pc indexed down to the selected points.
    │   ├── assert pc is a PointCloud
    │   ├── calls self._materialize_indices(device=pc.xyz.device)
    │   ├── impls indices = the materialized index tensor
    │   ├── assert every entry of indices is below pc.num_points
    │   ├── impls data: Dict[str, torch.Tensor] = {'xyz': pc.xyz indexed by indices}
    │   ├── if pc carries an indices field
    │   │   └── impls data['indices'] = pc.indices indexed by indices
    │   ├── else
    │   │   └── impls data['indices'] = indices
    │   ├── for each name in pc.field_names() past the first
    │   │   ├── if name == 'indices'
    │   │   │   └── continue
    │   │   └── impls data[name] = the field of pc under that name, indexed by indices
    │   ├── calls PointCloud(data=data)
    │   └── return  # the PointCloud wrapping data
    ├── def __str__(self) -> str
    │   ├── # Renders the selection, spelling the indices out only while there are at most five of them.
    │   ├── if self.indices is a list
    │   │   ├── impls num_indices = the length of self.indices
    │   │   ├── if num_indices is at most five
    │   │   │   └── return  # the indices spelled out
    │   │   └── return  # a stand-in naming num_indices
    │   ├── impls num_indices = the element count of self.indices
    │   ├── if num_indices is at most five
    │   │   └── return  # the indices spelled out
    │   └── return  # a stand-in naming num_indices
    └── def _materialize_indices(self, device: torch.device) -> torch.Tensor
        ├── # Turns the held indices into a non-negative int64 tensor sitting on the point cloud's device.
        ├── if self.indices is a list
        │   └── impls indices_tensor = self.indices as an int64 tensor on device
        ├── else
        │   ├── assert self.indices.dtype is torch.int64
        │   ├── assert the device of self.indices matches device
        │   └── impls indices_tensor = self.indices
        ├── assert every entry of indices_tensor is non-negative
        └── return indices_tensor
```

`data/structures/three_d/point_cloud/random_select.py`

```text
random_select.py
├── from typing import Any, Optional
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.select import Select
└── class RandomSelect
    ├── # Draws a random subset of a point cloud's points, sized either as a fraction of the cloud or as a fixed count.
    ├── def __init__(self, percentage: Optional[float] = None, count: Optional[int] = None) -> None
    │   ├── # Holds exactly one of the two sizing modes and leaves the other empty.
    │   ├── assert exactly one of percentage and count is given
    │   ├── if percentage is not None
    │   │   ├── assert percentage is an int or a float
    │   │   ├── assert percentage lies in (0, 1]
    │   │   ├── impls self.percentage = percentage as a float
    │   │   └── impls self.count = None
    │   └── else
    │       ├── assert count is an int
    │       ├── assert count is positive
    │       ├── impls self.count = count
    │       └── impls self.percentage = None
    ├── def __call__(self, pc: PointCloud, seed: Optional[Any] = None, generator: Optional[torch.Generator] = None) -> PointCloud
    │   ├── # Takes the sized random subset of pc, through a Select over the head of a random permutation of its point indices.
    │   ├── assert exactly one of seed and generator is given
    │   ├── assert pc is a PointCloud
    │   ├── impls device = the device of pc.xyz
    │   ├── impls num_points = pc.num_points
    │   ├── if generator is not None
    │   │   ├── assert the device type of generator matches that of device
    │   │   └── impls gen = generator
    │   ├── else
    │   │   ├── impls gen = a fresh torch.Generator on device
    │   │   ├── if seed is not an int
    │   │   │   ├── from utils.determinism.hash_utils import convert_to_seed
    │   │   │   ├── calls convert_to_seed(seed)
    │   │   │   └── impls seed = the int it returned
    │   │   └── impls gen is seeded with seed
    │   ├── if self.percentage is not None
    │   │   └── impls num_points_to_select = num_points scaled by self.percentage, truncated to an int
    │   ├── else
    │   │   └── impls num_points_to_select = the smaller of self.count and num_points  # impls-node-one-step:skip
    │   ├── impls indices = the leading num_points_to_select entries of a random permutation of num_points drawn from gen on device
    │   ├── calls Select(indices=indices)
    │   └── return  # the point cloud that selection hands back
    └── def __str__(self) -> str
        ├── # Renders the selection under whichever of the two sizing modes it carries.
        ├── if self.percentage is not None
        │   └── return  # a rendering naming self.percentage
        └── return  # a rendering naming self.count
```

`data/structures/three_d/point_cloud/__init__.py`

```text
__init__.py
├── from data.structures.three_d.point_cloud.io import load_point_cloud, save_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.random_select import RandomSelect
└── from data.structures.three_d.point_cloud.select import Select
```
