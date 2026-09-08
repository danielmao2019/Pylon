# Point Cloud Data Structure Code Structure

## Code structure trees

`data/structures/three_d/point_cloud/point_cloud.py`

```text
point_cloud.py
├── from typing import Any, Dict, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from utils.dtypes import COLOR_RANGE, CONCEPTUAL_NAME, NUMPY_DTYPE, TORCH_DTYPE, cast_lossless
└── class PointCloud
    ├── # One point cloud: named per-point fields, every one a torch tensor of the same length on one device, over one meta data record of what the construction source held.
    ├── # The meta data is a dict of one dict per field name, each holding both halves: the conceptual dtype that field's source held and the source columns it was assembled from. It is handed over whole or derived whole, once, at construction, and never again.
    ├── # The four underscore names below — _fields, _meta_data, _length, _device — are this class's own slots, and a bare one in any node means the slot on self; __setattr__ routes exactly those to the base setter and everything else to a validated field.
    ├── # What a field MEANS is the dtype its meta data entry holds, since that is the conceptual dtype its source held and is what a uint16 colour parked in an int32 tensor still is. A field the meta data does not name arrived after construction as a torch tensor, and torch holds no width it cannot name, so there its own dtype is exact.
    ├── def __init__(self, xyz: Optional[Union[np.ndarray, torch.Tensor]] = None, data: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None, meta_data: Optional[Dict[str, Dict[str, Any]]] = None, device: Optional[Union[str, torch.device]] = None) -> None
    │   ├── # Builds a point cloud from in-memory fields, deriving the record each field's source defines and applying over it whatever the meta data override states.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert xyz is None or xyz is an np.ndarray or a torch.Tensor
    │   │   ├── assert data is None or data is a dict whose keys are all str
    │   │   ├── if xyz is None
    │   │   │   ├── assert data is not None
    │   │   │   └── assert data carries 'xyz'  # coordinates arrive either on their own arg or inside data, and a construction naming them in neither is not a point cloud
    │   │   ├── else
    │   │   │   └── assert data is None or data carries no 'xyz'  # naming them in both leaves which one wins to the reader
    │   │   ├── assert meta_data is None or every value of it is a non-empty dict whose keys sit in ('dtype', 'layout')  # an entry states one half or both, the half it leaves out being the one the source defines
    │   │   ├── assert every 'layout' meta_data states is a non-empty tuple of distinct str
    │   │   └── assert device is None or device names a torch device
    │   ├── calls _validate_inputs()
    │   ├── def _normalize_inputs [local]
    │   │   ├── if xyz is None
    │   │   │   └── impls xyz = data['xyz']
    │   │   ├── impls data = xyz under the name 'xyz' followed by every other entry of data in its own order, or by nothing when data is None  # coordinates enter first, so field_names() reads coordinates-first without a splice
    │   │   ├── impls meta_data = meta_data when it is given, else an empty dict
    │   │   ├── impls device = device when it is given, else the device of data['xyz'] when it is a torch.Tensor, else the cpu device
    │   │   └── return data, meta_data, device
    │   ├── calls _normalize_inputs(xyz=xyz, data=data, meta_data=meta_data, device=device)
    │   ├── impls data, meta_data, device = the values it returned
    │   ├── def _derive_meta_data [local]
    │   │   ├── # Derives the record each field's own source defines and checks the override against it, producing the final meta data.
    │   │   ├── impls derived = an empty dict
    │   │   ├── for each name, value in data
    │   │   │   ├── impls entry = meta_data[name] when meta_data names this field, else an empty dict
    │   │   │   ├── impls source_dtype = CONCEPTUAL_NAME[the dtype of value]  # read before any cast, and kept whatever dtype the override states, since the record is what the source held
    │   │   │   ├── assert source_dtype is not 'uint64'  # uint64 is unsupported as a source dtype whatever the values are
    │   │   │   ├── impls layout = the 'layout' entry states, else a one-entry tuple of name  # an in-memory field is its own source and gets the identity mapping
    │   │   │   └── impls derived[name] = {'dtype': source_dtype, 'layout': layout}
    │   │   └── return derived
    │   ├── calls _derive_meta_data()
    │   ├── impls _meta_data = the record it derived  # resolved whole before the loop, since the rgb check below reads what a field means off it
    │   ├── def _apply_meta_data [local]
    │   │   ├── # Applies the override to the source data, casting each field the override states a dtype for.
    │   │   ├── for each name, entry in meta_data
    │   │   │   └── if entry states a dtype
    │   │   │       └── impls data[name] = data[name] cast to that dtype, in the system it arrived in  # the caller asked for it, so it converts as asked and whatever resolution it loses is the caller's own
    │   │   └── return data
    │   ├── calls _apply_meta_data()
    │   ├── impls data = the fields it applied the override to
    │   ├── impls _device = device
    │   ├── impls _length = the row count of data['xyz']
    │   ├── impls _fields = an empty dict
    │   └── for each name, value in data
    │       ├── calls self._assert_field_name_valid(name=name)
    │       ├── impls value_dtype = CONCEPTUAL_NAME[the dtype of value]  # what the value is carried as after the override, which is the storage question and not what the record says the field means
    │       ├── if value is an np.ndarray
    │       │   ├── calls cast_lossless(value, NUMPY_DTYPE[value_dtype])
    │       │   └── impls value = the array it cast, handed to torch  # the crossing into torch is this class's own decision rather than any caller's, so uint16 widens to int32 and a float128 column needing its width aborts here
    │       ├── impls tensor = value moved to self._device
    │       ├── calls self._validate_field(name=name, value=tensor)
    │       └── impls _fields[name] = tensor
    ├── @property def device(self) -> torch.device
    │   ├── # Hands back the one device every field of this point cloud sits on.
    │   └── return self._device
    ├── @property def num_points(self) -> int
    │   ├── # Hands back the number of points every field carries.
    │   └── return self._length
    ├── @property def meta_data(self) -> Dict[str, Dict[str, Any]]
    │   ├── # Hands back the meta data this point cloud was constructed with: one entry per source field, each holding the conceptual dtype that field's source held and the source columns it was assembled from.
    │   └── return self._meta_data
    ├── def field_names(self) -> Tuple[str, ...]
    │   ├── # Hands back every field name this point cloud carries, coordinates first because they entered first.
    │   ├── impls names = the keys of self._fields as a tuple
    │   └── return names
    ├── def __len__(self) -> int
    │   ├── # Serves the point count to len(), so a point cloud measures as its number of points.
    │   └── return self._length
    ├── def __getattr__(self, name: str) -> torch.Tensor
    │   ├── # Serves any field as an attribute under its own name, coordinates included, for a name ordinary attribute lookup did not find.
    │   ├── assert each of '_fields', '_meta_data', '_length' and '_device' sits in self.__dict__
    │   ├── if name sits in self._fields
    │   │   └── return self._fields[name]
    │   └── raise AttributeError  # the name is no field this point cloud carries
    ├── def __setattr__(self, name: str, value: object) -> None
    │   ├── # Routes an assignment to the private slot for an underscore name, and to a validated field otherwise, leaving the meta data exactly as construction wrote it.
    │   ├── # The value is only a tensor on the field branch; the slot branch carries the dicts, the length and the device this class stores about itself.
    │   ├── if name starts with '_'
    │   │   ├── impls the value goes to the slot through the base class attribute setter
    │   │   └── return
    │   ├── calls self._assert_field_name_valid(name=name)
    │   ├── calls self._validate_field(name=name, value=value)
    │   └── impls _fields[name] = value  # the meta data is left exactly as construction wrote it, and what the field now means follows from the two
    ├── def __delattr__(self, name: str) -> None
    │   ├── # Removes a field, leaving the meta data exactly as construction wrote it.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert name is not 'xyz'  # a point cloud without coordinates is not one
    │   │   └── assert name sits in self._fields
    │   ├── calls _validate_inputs()
    │   └── impls the entry under name leaves self._fields  # the meta data goes on naming the departed field, and save simply writes no column for one the obj no longer holds
    ├── def __getstate__(self) -> dict
    │   ├── # Hands the four private slots to pickle, so a point cloud and its meta data survive a round trip across a process boundary.
    │   ├── impls state = the four private slots _fields, _meta_data, _length and _device keyed by their own names  # impls-node-one-step:skip
    │   └── return state
    ├── def __setstate__(self, state: dict) -> None
    │   ├── # Restores the four private slots from a pickled state dict.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert state is a dict
    │   │   └── assert state carries each of '_fields', '_meta_data', '_length' and '_device'  # a payload written before the meta data existed carries no such slot and is refused here, to be regenerated rather than accepted through a shim
    │   ├── calls _validate_inputs()
    │   ├── impls _fields = state['_fields']
    │   ├── impls _meta_data = state['_meta_data']
    │   ├── impls _length = state['_length']
    │   └── impls _device = state['_device']
    ├── def _validate_field(self, name: str, value: torch.Tensor) -> None
    │   ├── # Checks one field's tensor-ness, rank, length and device, then the extra rules the names xyz and rgb carry.
    │   ├── assert value is a torch.Tensor
    │   ├── assert value is at least one-dimensional
    │   ├── assert value carries at least one point
    │   ├── assert the length of value matches self._length
    │   ├── assert the device of value matches self._device
    │   ├── if name == 'xyz'
    │   │   └── calls self.validate_xyz_tensor(value)
    │   └── elif name == 'rgb'
    │       ├── if name sits in self._meta_data
    │       │   └── impls colour_dtype = the 'dtype' of self._meta_data[name]  # the meta data is what says an int32 tensor holds a uint16 colour
    │       ├── else
    │       │   └── impls colour_dtype = CONCEPTUAL_NAME[the dtype of value]  # a colour assigned after construction is a torch tensor, and torch holds no width it cannot name
    │       └── calls self.validate_rgb_tensor(value, colour_dtype)
    ├── @staticmethod def validate_xyz_tensor(xyz: torch.Tensor) -> None
    │   ├── # Checks coordinates are an [N, 3] floating point tensor of any width, free of NaN and Inf.
    │   ├── assert xyz is a torch.Tensor
    │   ├── assert xyz is two-dimensional
    │   ├── assert xyz has three columns
    │   ├── assert xyz is a floating point tensor
    │   ├── assert xyz carries no NaN
    │   └── assert xyz carries no Inf
    ├── @staticmethod def validate_rgb_tensor(rgb: torch.Tensor, current_dtype: str) -> None
    │   ├── # Checks colors are an [N, 3] tensor whose values sit inside the range of the colour convention the dtype they MEAN names, which is not the range of the tensor parking them.
    │   ├── assert rgb is a torch.Tensor
    │   ├── assert rgb is two-dimensional
    │   ├── assert rgb has three columns
    │   ├── assert rgb carries no NaN
    │   ├── assert rgb carries no Inf
    │   ├── assert current_dtype sits in COLOR_RANGE  # a bool or int32 colour names no convention at all, and its absence from the table is what refuses it
    │   └── assert every value of rgb sits inside the bounds COLOR_RANGE[current_dtype] gives  # the conventions are told apart by dtype, never by inspecting the values, so a uint16 colour parked in an int32 tensor is bounded by uint16's range rather than int32's
    └── def _assert_field_name_valid(self, name: str) -> None
        ├── # Checks a field name is a str, is not underscore-prefixed, and collides with none of the reserved attribute names.
        ├── assert name is a str
        ├── assert name does not start with '_'
        └── assert name is none of 'device', 'num_points', 'meta_data', 'field_names', 'validate_xyz_tensor' and 'validate_rgb_tensor'  # a field under a name the class already binds would be written and then never readable
```

`data/structures/three_d/point_cloud/select.py`

```text
select.py
├── from typing import List, Union
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class Select
    ├── # Indexes a point cloud down to the points a fixed index list or index tensor names.
    ├── def __init__(self, indices: Union[torch.Tensor, List[int]]) -> None
    │   ├── # Holds the indices this selection will take, in the list or tensor form it was given.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert indices is a torch.Tensor or a list
    │   │   └── if indices is a list
    │   │       └── assert every entry of indices is an int  # a tensor's dtype and device are checked where the point cloud's device is known, which is not here
    │   ├── calls _validate_inputs()
    │   └── impls self.indices = indices
    ├── def __call__(self, pc: PointCloud) -> PointCloud
    │   ├── # Builds a new point cloud carrying every field of pc indexed down to the selected points, the meta data travelling across whole and each field's current dtype with it.
    │   ├── def _validate_inputs [local]
    │   │   └── assert pc is a PointCloud  # a reusable door, so it asserts what it needs whoever calls it
    │   ├── calls _validate_inputs()
    │   ├── calls self._materialize_indices(device=pc.device)
    │   ├── impls indices = the materialized index tensor
    │   ├── assert every entry of indices is below pc.num_points
    │   ├── impls fields = an empty dict
    │   ├── for each name in pc.field_names()
    │   │   └── impls fields[name] = the field of pc under that name, indexed by indices  # an indices field the cloud already carries is data like any other here, since the indices this selection takes are its own
    │   ├── if 'indices' does not sit in fields
    │   │   └── impls fields['indices'] = indices  # this selection made the field, so no source column stands behind it and the meta data crossing over names it nowhere
    │   ├── calls PointCloud(data=fields, meta_data=pc.meta_data)  # a selection is not a source, so the record crosses whole rather than being derived again from the indexed tensors
    │   └── return  # the point cloud it built
    ├── def __str__(self) -> str
    │   ├── # Renders the selection, spelling the indices out only while there are at most five of them.
    │   ├── impls num_indices = the length of self.indices when it is a list, else its element count
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
├── from utils.determinism.hash_utils import convert_to_seed
└── class RandomSelect
    ├── # Draws a random subset of a point cloud's points, sized either as a fraction of the cloud or as a fixed count.
    ├── def __init__(self, percentage: Optional[float] = None, count: Optional[int] = None) -> None
    │   ├── # Holds exactly one of the two sizing modes and leaves the other empty.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert exactly one of percentage and count is given
    │   │   ├── if percentage is not None
    │   │   │   ├── assert percentage is an int or a float
    │   │   │   └── assert percentage lies in (0, 1]
    │   │   └── else
    │   │       ├── assert count is an int
    │   │       └── assert count is positive
    │   ├── calls _validate_inputs()
    │   ├── if percentage is not None
    │   │   ├── impls self.percentage = percentage as a float
    │   │   └── impls self.count = None
    │   └── else
    │       ├── impls self.count = count
    │       └── impls self.percentage = None
    ├── def __call__(self, pc: PointCloud, seed: Optional[Any] = None, generator: Optional[torch.Generator] = None) -> PointCloud
    │   ├── # Takes the sized random subset of pc, through a Select over the head of a random permutation of its point indices.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert pc is a PointCloud
    │   │   └── assert exactly one of seed and generator is given  # the two randomness sources are one arg apiece, so the pair is checked once the second of them is reached
    │   ├── calls _validate_inputs()
    │   ├── impls device = pc.device
    │   ├── impls num_points = pc.num_points
    │   ├── if generator is not None
    │   │   ├── assert the device type of generator matches that of device
    │   │   └── impls gen = generator
    │   ├── else
    │   │   ├── impls gen = a fresh torch.Generator on device
    │   │   ├── if seed is not an int
    │   │   │   ├── calls convert_to_seed(seed)
    │   │   │   └── impls seed = the int it returned
    │   │   └── impls gen is seeded with seed
    │   ├── if self.percentage is not None
    │   │   └── impls num_points_to_select = num_points scaled by self.percentage, truncated to an int
    │   ├── else
    │   │   └── impls num_points_to_select = the smaller of self.count and num_points  # impls-node-one-step:skip
    │   ├── impls indices = the leading num_points_to_select entries of a random permutation of num_points drawn from gen on device
    │   ├── calls Select(indices=indices)
    │   ├── impls selected = that selection applied to pc  # constructing the Select is not applying it, and the applied result is what this returns
    │   └── return selected
    └── def __str__(self) -> str
        ├── # Renders the selection under whichever of the two sizing modes it carries.
        ├── if self.percentage is not None
        │   └── return  # a rendering naming self.percentage
        └── return  # a rendering naming self.count
```

`data/structures/three_d/point_cloud/__init__.py`

```text
__init__.py
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.random_select import RandomSelect
└── from data.structures.three_d.point_cloud.select import Select
```
