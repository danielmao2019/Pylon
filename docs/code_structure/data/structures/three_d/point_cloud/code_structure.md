# Point Cloud Data Structure Code Structure

## Code structure trees

`data/structures/three_d/point_cloud/point_cloud.py`

```text
point_cloud.py
├── from typing import Any, Dict, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from utils.dtypes import COLOR_RANGE, CONCEPTUAL_NAME, TORCH_DTYPE, cast_lossless, conceptual_name_of, convert_color_convention
└── class PointCloud
    ├── # One point cloud: named per-point fields, every one a torch tensor of the same length on one device, beside the record of what each of its source columns held and the target its fields stand on.
    ├── # The record is keyed on the source columns, holding for each the conceptual dtype that column held and the field it was assembled into when the cloud was constructed, and nothing after construction changes it.
    ├── # The target is keyed on the fields, holding for each the conceptual dtype it means and the columns it is assembled from, and every application of meta data replaces it whole.
    ├── # The five underscore names below — _fields, _meta_data, _target, _length, _device — are this class's own slots, and a bare one in any node means the slot on self; __setattr__ routes exactly those to the base setter and everything else to a validated field.
    ├── def __init__(self, xyz: Optional[Union[np.ndarray, torch.Tensor]] = None, data: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None, meta_data: Optional[Dict[str, Dict[str, Any]]] = None, device: Optional[Union[str, torch.device]] = None) -> None
    │   ├── # Builds a point cloud from the source's own columns, recording what each column held and the field it is assembled into, and then brings the fields onto the target the construction's meta data resolves into.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert xyz is None or xyz is an np.ndarray or a torch.Tensor
    │   │   ├── assert xyz is None or CONCEPTUAL_NAME[the dtype of xyz] is not 'uint64'  # uint64 is unsupported as a source dtype whatever the values are
    │   │   ├── assert data is None or data is a dict whose keys are all str and whose values are all np.ndarray or torch.Tensor
    │   │   ├── assert data is None or no value of data carries a uint64 dtype  # the same refusal for the columns handed in through data
    │   │   ├── assert xyz is not None or data is not None
    │   │   ├── assert xyz is None or data is None or 'xyz' does not sit in data  # the coordinates arg becomes one more column under that name, so a data entry already holding it would be overwritten without a word
    │   │   ├── assert meta_data is None or meta_data is a dict whose keys are all str and whose values are all dicts  # the record reads the layouts it states before apply_meta_data checks the rest at its own door
    │   │   ├── assert no dtype meta_data states is 'uint64'  # an override at this door is refused a uint64 the way a source is
    │   │   └── assert device is None or device names a torch device
    │   ├── calls _validate_inputs()
    │   ├── def _normalize_inputs [local]
    │   │   ├── if xyz is not None
    │   │   │   └── impls data = xyz under the name 'xyz' followed by every entry of data in its own order, or by nothing when data is None  # the coordinates arg is one more source column, so the two ways of handing them in are one dict from here on
    │   │   ├── impls device = device when it is given, else the device of the first value of data when it is a torch.Tensor, else the cpu device
    │   │   ├── if device names cuda without an index
    │   │   │   └── impls device = the cuda device carrying the index cuda is currently on  # a field lands on the current cuda device whatever index the name leaves out, so the slot names the index every field is checked against
    │   │   └── return data, device
    │   ├── calls _normalize_inputs(xyz=xyz, data=data, device=device)
    │   ├── impls data, device = the values it returned
    │   ├── impls _device = device
    │   ├── impls _length = the row count of the first value of data
    │   ├── impls _fields = each column of data raised to two dimensions, under its own name  # the source's columns exactly as the source gave them, which is what apply_meta_data reads them back out of; the one cast is the one it makes onto the target, and for a field no override names that target is the source dtype and so is also the crossing into torch
    │   ├── def _build_meta_data [local]
    │   │   ├── impls column_dtypes = CONCEPTUAL_NAME of each column of data, keyed by that column's own name  # read off the SOURCE columns, since uint16, uint32 and float128 reach torch only in the width TORCH_DTYPE parks them in, where their own names are gone
    │   │   ├── impls record = an empty dict
    │   │   ├── for each group in COORDINATE_COLUMN_NAMES
    │   │   │   └── if record names no coordinate field and every name in group sits in column_dtypes
    │   │   │       └── impls record['xyz'] = {'layout': group}  # what a source calls its coordinates is a fact about column names, so it is read here rather than told to this class by whoever loaded the file
    │   │   ├── for each group in COLOR_COLUMN_NAMES
    │   │   │   └── if record names no colour field and every name in group sits in column_dtypes
    │   │   │       └── impls record['rgb'] = {'layout': group}
    │   │   ├── for each name in column_dtypes
    │   │   │   └── if name sits in no layout record states
    │   │   │       └── impls record[name] = {'layout': a one-entry tuple of name}  # a column neither group claims stands for itself
    │   │   ├── for each entry in record
    │   │   │   ├── assert the column_dtypes of entry's layout columns are all one dtype  # columns that disagree abort rather than being promoted to a dtype that covers them all
    │   │   │   └── impls entry gains that dtype
    │   │   └── impls _meta_data = record  # the Layout Mapping over the source's own columns beside the dtype each column held, which is the record every later derivation reads and no field mutation ever rewrites
    │   ├── calls _build_meta_data()
    │   └── calls self.apply_meta_data(meta_data=meta_data)
    ├── def apply_meta_data(self, meta_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]
    │   ├── # Derives the target from the override written over the one the cloud stands on, brings the fields onto it, and hands it back.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert meta_data is None or meta_data is a dict whose keys are all str
    │   │   ├── assert every entry meta_data states is a dict whose keys all sit in ('dtype', 'layout')  # an override states one half or both, so a misspelled key names a half the design has no slot for
    │   │   ├── assert every entry meta_data states carries at least one of the two halves  # an entry asking for nothing would leave the target silently in force under a name the caller believes it changed
    │   │   ├── assert every dtype meta_data states sits in TORCH_DTYPE and is not 'uint64'
    │   │   ├── assert every layout meta_data states is a tuple of at least one str naming no column twice  # a caller-stated layout never passes through a record, so its own emptiness and distinctness are checked at the door it comes in by
    │   │   └── assert no column is named by two layouts meta_data states  # a column is assembled into one field and written under one name, so two layouts claiming it would each take it or overwrite each other
    │   ├── calls _validate_inputs()
    │   ├── def _normalize_inputs [local]
    │   │   ├── impls meta_data = an empty dict when meta_data is None
    │   │   └── return meta_data
    │   ├── calls _normalize_inputs(meta_data=meta_data)
    │   ├── impls meta_data = the value it returned
    │   ├── def _name_source_columns [local]
    │   │   ├── # Names every column the cloud's fields are assembled from beside the conceptual dtype it means, so a target layout regroups columns whatever field holds them now.
    │   │   ├── impls table = an empty dict
    │   │   ├── for each name, value in self._fields
    │   │   │   ├── impls columns = the 'layout' of self._target[name] when self._target names name, else a one-entry tuple of name  # a field assigned after the last application stands for its own whole block
    │   │   │   ├── assert no name in columns sits in table already  # a field named like another field's column would overwrite that column in silence
    │   │   │   ├── calls self.conceptual_dtype(name=name)
    │   │   │   ├── impls column_dtype = the dtype it named
    │   │   │   ├── if columns names exactly one column
    │   │   │   │   └── impls table[that column] = value beside column_dtype  # one name over a block of any width, which is how a pcd attribute and an in-memory field keep their columns together
    │   │   │   └── else
    │   │   │       ├── assert columns names exactly as many columns as value carries
    │   │   │       └── impls each column of value enters table under the name at its own position, beside column_dtype
    │   │   └── return table
    │   ├── calls _name_source_columns()
    │   ├── impls table = the columns it named
    │   ├── def _derive_target_meta_data [local]
    │   │   ├── # Writes the override over the target the cloud stands on, half by half, so every half the override leaves out is the one the fields already hold.
    │   │   ├── impls target = an empty dict
    │   │   ├── for each name in self._fields
    │   │   │   ├── impls columns = the 'layout' of self._target[name] when self._target names name, else a one-entry tuple of name
    │   │   │   ├── impls claimants = the keys of the meta_data entries whose layouts name a column of columns, in the order columns names those columns
    │   │   │   ├── if meta_data names name
    │   │   │   │   └── impls target[name] = a copy of the entry meta_data states for name
    │   │   │   ├── if claimants is empty and meta_data does not name name
    │   │   │   │   └── impls target[name] = {'layout': columns}  # a field the override leaves alone keeps the columns it is assembled from
    │   │   │   ├── for each claimant in claimants that target does not hold
    │   │   │   │   └── impls target[claimant] = a copy of the entry meta_data states for claimant  # a stated layout assembles the columns it names into its own field, which takes the place of the field those columns came out of
    │   │   │   └── if claimants is not empty
    │   │   │       └── for each column of columns that no layout meta_data states names
    │   │   │           └── impls target[column] = {'layout': a one-entry tuple of column}  # a column a claim leaves out stands for itself under its own name rather than vanishing with the field it came out of
    │   │   ├── for each name, entry meta_data states that target does not hold
    │   │   │   └── impls target[name] = a copy of entry
    │   │   ├── for each name, entry of target that states no layout
    │   │   │   └── impls entry gains the 'layout' of self._target[name] when self._target names name, else a one-entry tuple of name  # a lone dtype half keeps the columns the field is already assembled from
    │   │   ├── assert no column is named by two layouts of target  # a column is assembled into one field, so a lone dtype half for a field whose columns a stated layout claimed contradicts that claim
    │   │   ├── for each name, entry in target
    │   │   │   ├── if table names every column of entry's layout
    │   │   │   │   └── impls entry gains, when it states none, the dtype the first of those columns means in table  # columns that disagree are refused once the target is applied
    │   │   │   ├── elif table names no column of entry's layout
    │   │   │   │   ├── assert self._fields holds a field under name  # an entry reaching neither the cloud's own columns nor a field it holds names nothing at all, which is a misspelling rather than a field to drop in silence
    │   │   │   │   ├── assert entry's layout names exactly as many columns as that field carries, a one-dimensional field carrying one  # the layout names that field's own columns afresh, one name per column
    │   │   │   │   ├── calls self.conceptual_dtype(name=name)
    │   │   │   │   └── impls entry gains the dtype it named, when it states none
    │   │   │   └── else
    │   │   │       └── assert 0  # a layout is either columns the cloud holds or a field's own columns named afresh, never a mixture, and a half-matching one is a misspelling rather than either
    │   │   ├── impls target = the entry target holds under 'xyz' first when it holds one, followed by every other entry in its own order  # coordinates lead the fields whatever order the source's columns arrived in
    │   │   └── return target  # a cloud whose columns assemble into no coordinate field is legal here, a reader building one from a positional source having no coordinates to name yet
    │   ├── calls _derive_target_meta_data(table=table)
    │   ├── impls target = the meta data it derived
    │   ├── def _apply_target_meta_data [local]
    │   │   ├── # Rebuilds every field the target names out of the columns its layout names, on the convention and at the width that entry states.
    │   │   ├── impls fields = an empty dict
    │   │   ├── for each name, entry in target
    │   │   │   ├── if table names every column of entry's layout
    │   │   │   │   └── impls columns = each table column entry's layout names, a one-dimensional one raised to one column wide, beside the dtype it means, in the order the layout names them
    │   │   │   ├── else
    │   │   │   │   ├── calls self.conceptual_dtype(name=name)
    │   │   │   │   └── impls columns = the one block self._fields holds under name, beside the dtype it named  # the layout names this field's own columns afresh, so nothing is regrouped
    │   │   │   ├── if meta_data states a dtype for name
    │   │   │   │   └── for each value, column_dtype in columns
    │   │   │   │       ├── if name == 'rgb' and column_dtype and entry's dtype both sit in COLOR_RANGE and differ
    │   │   │   │       │   ├── calls convert_color_convention(values=value, source_dtype=column_dtype, target_dtype=entry's dtype)
    │   │   │   │       │   ├── impls converted = the colours it mapped, at TORCH_DTYPE[entry's dtype]  # the mapping hands back double precision, and a colour is held at the storage its own convention names
    │   │   │   │       │   ├── calls convert_color_convention(values=converted, source_dtype=entry's dtype, target_dtype=column_dtype)
    │   │   │   │       │   ├── assert what it mapped back, held at the storage of value, equals value  # lossless means the source values come back exactly on the source's own convention, and a float32 source's own convention is float32's grid rather than double's
    │   │   │   │       │   └── impls that column becomes converted, meaning entry's dtype
    │   │   │   │       └── else
    │   │   │   │           ├── calls cast_lossless(values=value, dtype=TORCH_DTYPE[entry's dtype])  # a narrowing the target cannot hold exactly aborts inside the cast, a caller wanting one narrowing its own values before handing them in
    │   │   │   │           └── impls that column becomes the tensor it cast, meaning entry's dtype
    │   │   │   ├── assert every column of columns means entry's dtype  # columns that still disagree once the target dtype is applied abort rather than being promoted to a dtype that covers them all
    │   │   │   └── impls fields[name] = the values of columns joined along the column axis  # three ply columns and one pcd attribute reach the same [N, 3]
    │   │   ├── impls _fields = fields
    │   │   └── impls _target = target  # the record is left exactly as construction wrote it, so what an application moves is the fields and the target they stand on
    │   ├── calls _apply_target_meta_data(table=table, target=target)
    │   ├── for each name, value in self._fields
    │   │   ├── calls self._assert_field_name_valid(name=name)
    │   │   └── calls self._validate_field(name=name, value=value)  # every slot is assigned by now, so a colour is bounded by the convention the target it was just made to match names
    │   └── return target  # the target is handed back for the save, whose reverse mapping reads the columns each field maps back to off it
    ├── @property def device(self) -> torch.device
    │   ├── # Hands back the one device every field of this point cloud sits on.
    │   └── return self._device
    ├── @property def num_points(self) -> int
    │   ├── # Hands back the number of points every field carries.
    │   └── return self._length
    ├── @property def meta_data(self) -> Dict[str, Dict[str, Any]]
    │   ├── # Hands back the meta data this point cloud carries: one entry per field, each holding the conceptual dtype that field's source held and the source columns it was assembled from.
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
    │   ├── # Routes an assignment to the private slot for an underscore name, and to a validated field otherwise, leaving the meta data exactly as it stands.
    │   ├── # The value is only a tensor on the field branch; the slot branch carries the dicts, the length and the device this class stores about itself.
    │   ├── if name starts with '_'
    │   │   ├── impls the value goes to the slot through the base class attribute setter
    │   │   └── return
    │   ├── calls self._assert_field_name_valid(name=name)
    │   ├── calls self._validate_field(name=name, value=value)
    │   └── impls _fields[name] = value  # an overwritten field goes on meaning what its target says while it sits at the storage that dtype names, which is how a cloned colour keeps its convention
    ├── def __delattr__(self, name: str) -> None
    │   ├── # Removes a field, leaving the record and the target exactly as they stand.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert name is not 'xyz'  # a point cloud without coordinates is not one
    │   │   └── assert name sits in self._fields
    │   ├── calls _validate_inputs()
    │   └── impls the entry under name leaves self._fields  # the record goes on naming the departed field's columns, and nothing is written for a field the obj no longer holds
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
    │       ├── impls color_dtype = the 'dtype' of self._meta_data[name] when the record names name, else CONCEPTUAL_NAME[the dtype of value]  # the record is what says an int32 tensor holds a uint16 colour, and a colour under a name the record never saw is exact in its own tensor
    │       └── calls self.validate_rgb_tensor(value, color_dtype)
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
        └── assert name is none of 'device', 'num_points', 'meta_data', 'field_names', 'apply_meta_data', 'validate_xyz_tensor' and 'validate_rgb_tensor'  # a field under a name the class already binds would be written and then never readable
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
