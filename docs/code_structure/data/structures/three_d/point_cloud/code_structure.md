# Point Cloud Data Structure Code Structure

## Code structure trees

`data/structures/three_d/point_cloud/point_cloud.py`

```text
point_cloud.py
├── from typing import Any, Dict, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from utils.dtypes import COLOR_RANGE, CONCEPTUAL_NAME, TORCH_DTYPE, cast_lossless, convert_color_convention
├── COORDINATE_COLUMN_NAMES  # the column-name groups a source calls its coordinates, in the order they are tried: ('x', 'y', 'z') as ply, las and off name them, ('positions',) as open3d does, and ('xyz',) as a caller handing one in-memory block in does
├── COLOR_COLUMN_NAMES  # the column-name groups a source calls its colours, in the order they are tried: ('red', 'green', 'blue') as ply and las name them, ('colors',) as open3d does, and ('rgb',) as a caller handing one in-memory block in does
└── class PointCloud
    ├── # One point cloud: named per-point fields, every one a torch tensor of the same length on one device, over one meta data entry per field of what that field's source held.
    ├── # A cloud is constructed out of the source's own columns under the Layout Mapping that source defines, and becomes the cloud a caller wanted only once apply_meta_data has run over the halves that source left for the caller to state.
    ├── # apply_meta_data runs once inside every construction and again on each load and save, so it names the source columns back out of the fields it has already assembled rather than assuming it meets them unassembled.
    ├── # The four underscore names below — _fields, _meta_data, _length, _device — are this class's own slots, and a bare one in any node means the slot on self; __setattr__ routes exactly those to the base setter and everything else to a validated field.
    ├── def __init__(self, xyz: Optional[Union[np.ndarray, torch.Tensor]] = None, data: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None, meta_data: Optional[Dict[str, Dict[str, Any]]] = None, device: Optional[Union[str, torch.device]] = None) -> None
    │   ├── # Builds a point cloud from the source's own columns, recording what each column held or standing the record it is handed in place of that, and then bringing the fields onto it.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert xyz is None or xyz is an np.ndarray or a torch.Tensor
    │   │   ├── assert xyz is None or CONCEPTUAL_NAME[the dtype of xyz] is not 'uint64'  # uint64 is unsupported as a source dtype whatever the values are
    │   │   ├── assert data is None or data is a dict whose keys are all str
    │   │   ├── assert data is None or no value of data carries a uint64 dtype  # the same refusal for the columns handed in through data
    │   │   ├── assert xyz is not None or data is not None
    │   │   ├── assert xyz is None or data is None or 'xyz' does not sit in data  # the coordinates arg becomes one more column under that name, so a data entry already holding it would be overwritten without a word
    │   │   ├── assert meta_data is None or meta_data is a dict whose keys are all str
    │   │   ├── assert every entry meta_data holds is a dict whose keys are exactly 'dtype' and 'layout'  # a record arrives resolved rather than half-stated, so a misspelled key, a lone half and an entry stating nothing are all refused here rather than read past
    │   │   ├── assert every dtype meta_data holds sits in TORCH_DTYPE and is not 'uint64'
    │   │   ├── assert every layout meta_data holds is a tuple of at least one str naming no column twice
    │   │   └── assert device is None or device names a torch device
    │   ├── calls _validate_inputs()
    │   ├── def _normalize_inputs [local]
    │   │   ├── if xyz is not None
    │   │   │   └── impls data = xyz under the name 'xyz' followed by every entry of data in its own order, or by nothing when data is None  # the coordinates arg is one more source column, so the two ways of handing them in are one dict from here on
    │   │   ├── impls device = device when it is given, else the device of the first value of data when it is a torch.Tensor, else the cpu device
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
    │   ├── # Derives the meta data this cloud should carry from the one it records and the override, makes the cloud match it, and hands the target back for the writer that has to write the file at it.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert meta_data is None or meta_data is a dict whose keys are all str
    │   │   ├── assert every entry meta_data states is a dict whose keys all sit in ('dtype', 'layout')  # an override states one half or both, so a misspelled key names a half the design has no slot for
    │   │   ├── assert every entry meta_data states carries at least one of the two halves  # an entry asking for nothing would leave the record silently in force under a name the caller believes it changed
    │   │   ├── assert every dtype meta_data states sits in TORCH_DTYPE and is not 'uint64'
    │   │   └── assert every layout meta_data states is a tuple of at least one str naming no column twice  # a caller-stated layout never passes through a record, so its own emptiness and distinctness are checked at the door it comes in by
    │   ├── calls _validate_inputs()
    │   ├── def _name_source_columns [local]
    │   │   ├── # Names every source column the cloud still holds beside what that column means, so a target layout regroups columns whatever field the record has already assembled them into.
    │   │   ├── impls table = an empty dict
    │   │   ├── for each name, value in self._fields
    │   │   │   ├── impls columns = the layout of self._meta_data[name] when the record names name, else a one-entry tuple of name  # a field assigned after construction stands outside the record and stands for its own whole block
    │   │   │   ├── impls column_dtype = the 'dtype' of self._meta_data[name] when the record names name, else CONCEPTUAL_NAME[the dtype of value]
    │   │   │   ├── if columns names exactly one column
    │   │   │   │   └── impls table[that column] = value beside column_dtype  # one name over a block of any width, which is how a pcd attribute and an in-memory field keep their columns together
    │   │   │   └── else
    │   │   │       ├── assert columns names exactly as many columns as value carries
    │   │   │       └── impls each column of value enters table under the name at its own position, beside column_dtype
    │   │   └── return table
    │   ├── calls _name_source_columns()
    │   ├── impls table = the columns it named
    │   ├── def _derive_target_meta_data [local]
    │   │   ├── # Writes the override over the record half by half, so every half the override leaves out is the one the record already defines.
    │   │   ├── if meta_data states a layout for at least one field and self._meta_data names no coordinate field
    │   │   │   ├── assert every entry meta_data states carries a layout  # a source that numbers its columns defines no field for a lone dtype half to name, so a misspelling fails here rather than standing silently
    │   │   │   └── impls target = each entry meta_data states  # a caller writing the layout by hand over such a source has chosen which columns become fields, so a column none of them names is simply absent
    │   │   ├── else
    │   │   │   ├── impls target = each entry of self._meta_data with the entry meta_data states for that field written over it, half by half  # the override outranks the record, which is how a ply caller restates one field and leaves the file's own layout standing for the rest
    │   │   │   ├── for each name, entry meta_data states that self._meta_data does not name
    │   │   │   │   ├── assert entry states a layout  # a field the record never saw defines neither half, so the caller supplies the one that says which columns it is
    │   │   │   │   └── impls target[name] = entry
    │   │   │   └── for each name, entry of self._meta_data that meta_data does not name
    │   │   │       └── if a layout meta_data states names any column of entry's layout
    │   │   │           └── impls that entry leaves target  # a caller-stated layout CONSUMES the source columns it names, so a column assembled into the field the caller asked for does not also survive under the field the record had assembled it into
    │   │   ├── for each name, entry of target that self._meta_data names and meta_data does not
    │   │   │   └── if table names no column of entry's layout and self._fields holds no field under name
    │   │   │       └── impls that entry leaves target  # the field was deleted, and save writes no column for one the obj no longer holds
    │   │   ├── for each name, value in the fields self._fields holds that target names nowhere
    │   │   │   └── if no layout target states names name
    │   │   │       └── impls target[name] = {'layout': a one-entry tuple of name, 'dtype': CONCEPTUAL_NAME[the dtype of value]}  # a field assigned after construction takes both halves from itself, while a column another entry's layout already assembles is that field's column rather than a field of its own
    │   │   ├── for each name, entry in target
    │   │   │   ├── if table names every column of entry's layout
    │   │   │   │   ├── assert the columns entry's layout names mean one dtype between them in table  # columns that disagree abort rather than being promoted to a dtype that covers them all
    │   │   │   │   └── impls entry gains that dtype when it states none  # a half the override leaves out is the one the record defines
    │   │   │   ├── elif table names no column of entry's layout
    │   │   │   │   ├── assert self._fields holds a field under name  # an entry reaching neither the cloud's own columns nor a field it holds names nothing at all, which is a misspelling rather than a field to drop in silence
    │   │   │   │   ├── if meta_data states a layout for name
    │   │   │   │   │   └── assert entry's layout names exactly as many columns as that field carries  # the reverse mapping writes one output column per name, so a count that disagrees leaves the writer with no name for a column
    │   │   │   │   └── impls entry gains the dtype self._meta_data holds for name, or the one its own field carries where the record names it nowhere, when it states none
    │   │   │   └── else
    │   │   │       └── assert 0  # a layout is either the source columns a field is assembled from or the names its columns are written out under, never a mixture, and a half-matching one is a misspelling rather than either
    │   │   └── return target  # a cloud whose columns assemble into no coordinate field is legal here, a reader building one from a positional source having no coordinates to name yet
    │   ├── calls _derive_target_meta_data(table=table)
    │   ├── impls target = the meta data it derived
    │   ├── def _apply_target_meta_data [local]
    │   │   ├── # Rebuilds every field the target names out of the columns its layout names, on the convention and at the width that entry states.
    │   │   ├── impls fields = an empty dict
    │   │   ├── for each name, entry in target
    │   │   │   ├── if table names every column of entry's layout
    │   │   │   │   ├── impls value = the table columns entry's layout names, joined along the column axis  # a one-dimensional column becomes one column wide and a block that is already two-dimensional keeps the width it has, which is what lets three ply columns and one pcd attribute reach the same [N, 3]
    │   │   │   │   └── impls source_dtype = the one dtype those table columns mean between them  # the dtype the source held, which the record keeps whatever the target asks the values to become
    │   │   │   ├── else
    │   │   │   │   ├── impls value = the field self._fields holds under name  # the layout is naming this field's output columns rather than selecting the cloud's own, so nothing is regrouped
    │   │   │   │   └── impls source_dtype = the 'dtype' of self._meta_data[name] when the record names name, else CONCEPTUAL_NAME[the dtype of value]
    │   │   │   ├── if meta_data states a dtype for name
    │   │   │   │   ├── if name == 'rgb' and entry's dtype is not source_dtype
    │   │   │   │   │   ├── calls convert_color_convention(values=value, source_dtype=source_dtype, target_dtype=entry's dtype)
    │   │   │   │   │   ├── impls converted = the colours it mapped onto the target convention
    │   │   │   │   │   ├── impls converted = converted at TORCH_DTYPE[entry's dtype]  # the mapping hands back double precision, and a colour is held at the storage its own convention names
    │   │   │   │   │   ├── calls convert_color_convention(values=converted, source_dtype=entry's dtype, target_dtype=source_dtype)
    │   │   │   │   │   ├── assert what it mapped back equals value  # a colour conversion is lossless when the source values come back exactly, which is a question about the two conventions and never about the width holding them
    │   │   │   │   │   ├── impls value = converted
    │   │   │   │   │   └── impls source_dtype = entry's dtype  # the colours sit on the target's range now, so that is the convention they MEAN and the one the record has to name for the next reader to read them by
    │   │   │   │   └── else
    │   │   │   │       ├── calls cast_lossless(values=value, dtype=TORCH_DTYPE[entry's dtype])
    │   │   │   │       └── impls value = the tensor it cast  # a narrowing the target cannot hold exactly aborts inside the cast, a caller wanting one narrowing its own values before handing them in
    │   │   │   └── impls fields[name] = value  # a target dtype the caller did not state is what the field already means, so nothing is cast on the record's account and a record naming one width over a tensor of another leaves that tensor where it is
    │   │   └── impls _fields = fields  # the record is left exactly as construction wrote it, so what an override moves is the fields and the target this hands back, never the provenance
    │   ├── calls _apply_target_meta_data(table=table, target=target)
    │   ├── for each name, value in self._fields
    │   │   ├── calls self._assert_field_name_valid(name=name)
    │   │   └── calls self._validate_field(name=name, value=value)  # both slots are assigned by now, so a colour is bounded by the convention the record it was just made to match names
    │   └── return target  # the dtype half a caller states never reaches the record, so the target is handed back for the writer that has to write the file at it
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
    │   └── impls _fields[name] = value  # the meta data is left exactly as it stands, and what the field now means follows from the two
    ├── def __delattr__(self, name: str) -> None
    │   ├── # Removes a field, leaving the meta data exactly as it stands.
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
