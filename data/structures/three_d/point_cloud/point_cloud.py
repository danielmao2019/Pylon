from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from utils.dtypes import (
    COLOR_RANGE,
    CONCEPTUAL_NAME,
    TORCH_DTYPE,
    cast_lossless,
    convert_color_convention,
)

# the column-name groups a source calls its coordinates, in the order they are tried: ('x', 'y', 'z') as ply, las and off name them, ('positions',) as open3d does, and ('xyz',) as a caller handing one in-memory block in does
COORDINATE_COLUMN_NAMES = (('x', 'y', 'z'), ('positions',), ('xyz',))

# the column-name groups a source calls its colours, in the order they are tried: ('red', 'green', 'blue') as ply and las name them, ('colors',) as open3d does, and ('rgb',) as a caller handing one in-memory block in does
COLOR_COLUMN_NAMES = (('red', 'green', 'blue'), ('colors',), ('rgb',))


class PointCloud:
    """One point cloud: named per-point fields, every one a torch tensor of the same length on one device, over one meta data entry per field of what that field's source held.

    A cloud is constructed out of the source's own columns under the Layout Mapping that source defines, or out of another cloud's fields under the record that comes with them, and becomes the cloud a caller wanted only once a load or a save has run apply_meta_data over the halves that source left for the caller to state.

    apply_meta_data runs once inside every construction and again on each load and save, so it names the source columns back out of the fields it has already assembled rather than assuming it meets them unassembled.

    The four underscore names below - _fields, _meta_data, _length, _device - are this class's own slots, and a bare one in any node means the slot on self; __setattr__ routes exactly those to the base setter and everything else to a validated field.
    """

    def __init__(
        self,
        xyz: Optional[Union[np.ndarray, torch.Tensor]] = None,
        data: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        meta_data: Optional[Dict[str, Dict[str, Any]]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Builds a point cloud from the source's own columns, recording what each column held or standing the record it is handed in place of that, and then bringing the fields onto it.

        Args:
            xyz: The coordinate columns, as a numpy array or a torch tensor of shape [N, 3] carrying any floating point dtype, or None when the coordinates arrive as an entry of data.
            data: The source's own columns keyed by the name the source gives each, every value a numpy array or a torch tensor of shape [N] or [N, C] carrying any dtype this repo's conceptual dtype table names except uint64, or None when xyz is the only column.
            meta_data: The record another point cloud hands over, which is also what this construction applies over the columns it was handed, one entry per field keyed by field name, each entry holding a 'dtype' naming a conceptual dtype and a 'layout' naming the source columns that field is assembled from, or None when the record is to be built from the columns themselves.
            device: The torch device every field is to sit on, as a string or a torch.device, or None to take the device of the first column when it is already a torch tensor and the cpu device otherwise.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert xyz is None or isinstance(
                xyz, (np.ndarray, torch.Tensor)
            ), f"coordinates arrive as a numpy array or a torch tensor: type(xyz)={type(xyz)}"
            assert (
                xyz is None or CONCEPTUAL_NAME[xyz.dtype] != 'uint64'
            ), f"uint64 is unsupported as a source dtype whatever the values are: xyz.dtype={None if xyz is None else xyz.dtype}"
            assert data is None or isinstance(
                data, dict
            ), f"the source's columns arrive as a dict: type(data)={type(data)}"
            assert data is None or all(
                isinstance(name, str) for name in data.keys()
            ), f"every source column is named by a str: data keys={None if data is None else tuple(data.keys())}"
            assert data is None or all(
                isinstance(column, (np.ndarray, torch.Tensor))
                for column in data.values()
            ), f"every source column arrives as a numpy array or a torch tensor: column types={None if data is None else {name: type(column) for name, column in data.items()}}"
            assert data is None or all(
                CONCEPTUAL_NAME[column.dtype] != 'uint64' for column in data.values()
            ), f"uint64 is unsupported as a source dtype whatever the values are: column dtypes={None if data is None else {name: column.dtype for name, column in data.items()}}"
            assert (
                xyz is not None or data
            ), f"a point cloud is built from coordinates, from columns, or from both, never from neither: xyz={xyz}, data={data}"
            assert (
                xyz is None or data is None or 'xyz' not in data
            ), f"the coordinates arg becomes one more column under the name 'xyz', so a data entry already holding it would be overwritten without a word: data keys={None if data is None else tuple(data.keys())}"
            assert meta_data is None or isinstance(
                meta_data, dict
            ), f"a record arrives as a dict: type(meta_data)={type(meta_data)}"
            assert meta_data is None or all(
                isinstance(name, str) for name in meta_data.keys()
            ), f"every field a record names is named by a str: meta_data keys={None if meta_data is None else tuple(meta_data.keys())}"
            for name, entry in (meta_data or {}).items():
                assert isinstance(entry, dict) and set(entry.keys()) == {
                    'dtype',
                    'layout',
                }, f"a record arrives resolved rather than half-stated, so a misspelled key, a lone half and an entry stating nothing are all refused here rather than read past: name={name}, entry={entry}"
                assert (
                    entry['dtype'] in TORCH_DTYPE and entry['dtype'] != 'uint64'
                ), f"a recorded dtype names a conceptual dtype torch stores and is never uint64: name={name}, dtype={entry['dtype']}, dtypes torch stores={sorted(TORCH_DTYPE.keys())}"
                assert (
                    isinstance(entry['layout'], tuple)
                    and len(entry['layout']) >= 1
                    and all(isinstance(column, str) for column in entry['layout'])
                    and len(set(entry['layout'])) == len(entry['layout'])
                ), f"a recorded layout is a tuple of at least one str naming no column twice: name={name}, layout={entry['layout']}"
            assert device is None or isinstance(
                device, (str, torch.device)
            ), f"a device is named by a str or a torch.device: type(device)={type(device)}, device={device}"

        _validate_inputs()

        def _normalize_inputs(
            xyz: Optional[Union[np.ndarray, torch.Tensor]],
            data: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]],
            device: Optional[Union[str, torch.device]],
        ) -> Tuple[
            Dict[str, Union[np.ndarray, torch.Tensor]],
            torch.device,
        ]:
            if xyz is not None:
                # the coordinates arg is one more source column, so the two ways of handing them in are one dict from here on
                data = {'xyz': xyz, **(data if data is not None else {})}

            if device is not None:
                device = torch.device(device)
            elif isinstance(next(iter(data.values())), torch.Tensor):
                device = next(iter(data.values())).device
            else:
                device = torch.device('cpu')
            if device.type == 'cuda' and device.index is None:
                # a field lands on the current cuda device whatever index the name leaves out, so the slot names that same index rather than the bare type
                device = torch.device('cuda', torch.cuda.current_device())

            return data, device

        data, device = _normalize_inputs(xyz=xyz, data=data, device=device)

        self._device = device
        self._length = next(iter(data.values())).shape[0]
        # the source's columns, not yet the fields a record names
        fields = {}
        for name, column in data.items():
            value = cast_lossless(
                values=column, dtype=TORCH_DTYPE[CONCEPTUAL_NAME[column.dtype]]
            ).to(self._device)
            fields[name] = value if value.ndim >= 2 else value.unsqueeze(-1)
        self._fields = fields

        def _build_meta_data() -> None:
            """Records what each of the source's own columns held, or stands the record this construction is handed in place of that.

            Args:
                None.

            Returns:
                None.
            """
            if meta_data is not None:
                # a record handed over is already resolved, so it stands as this cloud's own rather than being derived a second time from the tensors it comes with
                self._meta_data = {
                    name: dict(entry) for name, entry in meta_data.items()
                }
                return

            # read off the SOURCE columns, since uint16, uint32 and float128 reach torch only in the width TORCH_DTYPE parks them in, where their own names are gone
            column_dtypes = {
                name: CONCEPTUAL_NAME[column.dtype] for name, column in data.items()
            }
            record = {}
            for group in COORDINATE_COLUMN_NAMES:
                if 'xyz' not in record and all(name in column_dtypes for name in group):
                    # what a source calls its coordinates is a fact about column names, so it is read here rather than told to this class by whoever loaded the file
                    record['xyz'] = {'layout': group}
            for group in COLOR_COLUMN_NAMES:
                if 'rgb' not in record and all(name in column_dtypes for name in group):
                    record['rgb'] = {'layout': group}
            for name in column_dtypes:
                if not any(name in entry['layout'] for entry in record.values()):
                    # a column neither group claims stands for itself
                    record[name] = {'layout': (name,)}
            for name, entry in record.items():
                dtypes = set(column_dtypes[column] for column in entry['layout'])
                assert (
                    len(dtypes) == 1
                ), f"columns that disagree abort rather than being promoted to a dtype that covers them all: name={name}, layout={entry['layout']}, column dtypes={ {column: column_dtypes[column] for column in entry['layout']} }"
                entry['dtype'] = next(iter(dtypes))
            # the Layout Mapping over the source's own columns beside the dtype each column held, which is the record every later derivation reads and no field mutation ever rewrites
            self._meta_data = record

        _build_meta_data()
        self.apply_meta_data(meta_data=meta_data)

    def apply_meta_data(
        self, meta_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Derives the meta data this cloud should carry from the one it records and the override, makes the cloud match it, and hands the target back for the writer that has to write the file at it.

        Args:
            meta_data: The override, one entry per field keyed by field name, each entry stating a 'dtype' naming a conceptual dtype, a 'layout' naming source columns as a tuple of str, or both, or None to apply the record with nothing written over it.

        Returns:
            The target meta data, one entry per field keyed by field name, each entry holding a 'dtype' naming a conceptual dtype and a 'layout' naming a tuple of source column names.
        """

        def _validate_inputs() -> None:
            assert meta_data is None or isinstance(
                meta_data, dict
            ), f"an override arrives as a dict: type(meta_data)={type(meta_data)}"
            assert meta_data is None or all(
                isinstance(name, str) for name in meta_data.keys()
            ), f"every field an override states is named by a str: meta_data keys={None if meta_data is None else tuple(meta_data.keys())}"
            for name, entry in (meta_data or {}).items():
                assert isinstance(entry, dict) and all(
                    key in ('dtype', 'layout') for key in entry.keys()
                ), f"an override states one half or both, so a misspelled key names a half the design has no slot for: name={name}, entry={entry}"
                assert (
                    len(entry) >= 1
                ), f"an entry asking for nothing would leave the record silently in force under a name the caller believes it changed: name={name}, entry={entry}"
                assert 'dtype' not in entry or (
                    entry['dtype'] in TORCH_DTYPE and entry['dtype'] != 'uint64'
                ), f"a stated dtype names a conceptual dtype torch stores and is never uint64: name={name}, dtype={entry.get('dtype')}, dtypes torch stores={sorted(TORCH_DTYPE.keys())}"
                assert 'layout' not in entry or (
                    isinstance(entry['layout'], tuple)
                    and len(entry['layout']) >= 1
                    and all(isinstance(column, str) for column in entry['layout'])
                    and len(set(entry['layout'])) == len(entry['layout'])
                ), f"a caller-stated layout never passes through a record, so its own emptiness and distinctness are checked at the door it comes in by: name={name}, layout={entry.get('layout')}"

        _validate_inputs()

        def _name_source_columns() -> Dict[str, Tuple[torch.Tensor, str]]:
            """Names every source column the cloud still holds beside what that column means, so a target layout regroups columns whatever field the record has already assembled them into.

            Args:
                None.

            Returns:
                The source columns keyed by column name, each value a pair of the column's values as a torch tensor of shape [N] or [N, C] and the conceptual dtype name that column means.
            """
            table = {}
            for name, value in self._fields.items():
                # a field assigned after construction stands outside the record and stands for its own whole block
                columns = (
                    self._meta_data[name]['layout']
                    if name in self._meta_data
                    else (name,)
                )
                column_dtype = (
                    self._meta_data[name]['dtype']
                    if name in self._meta_data
                    else CONCEPTUAL_NAME[value.dtype]
                )
                if len(columns) == 1:
                    # one name over a block of any width, which is how a pcd attribute and an in-memory field keep their columns together
                    table[columns[0]] = (value, column_dtype)
                else:
                    assert (
                        value.ndim >= 2 and len(columns) == value.shape[1]
                    ), f"a multi-column layout names exactly as many columns as the field carries: name={name}, layout={columns}, value.shape={tuple(value.shape)}"
                    for index, column in enumerate(columns):
                        table[column] = (value[:, index], column_dtype)
            return table

        table = _name_source_columns()

        def _derive_target_meta_data(
            table: Dict[str, Tuple[torch.Tensor, str]],
        ) -> Dict[str, Dict[str, Any]]:
            """Writes the override over the record half by half, so every half the override leaves out is the one the record already defines.

            Args:
                table: The source columns keyed by column name, each value a pair of the column's values as a torch tensor of shape [N] or [N, C] and the conceptual dtype name that column means.

            Returns:
                The target meta data, one entry per field keyed by field name, each entry holding a 'dtype' naming a conceptual dtype and a 'layout' naming a tuple of source column names.
            """
            if (
                meta_data is not None
                and any('layout' in entry for entry in meta_data.values())
                and 'xyz' not in self._meta_data
            ):
                for name, entry in meta_data.items():
                    assert (
                        'layout' in entry
                    ), f"a source that numbers its columns defines no field for a lone dtype half to name, so a misspelling fails here rather than standing silently: name={name}, entry={entry}, recorded fields={tuple(self._meta_data.keys())}"
                # a caller writing the layout by hand over such a source has chosen which columns become fields, so a column none of them names is simply absent
                target = {name: dict(entry) for name, entry in meta_data.items()}
            else:
                # the override outranks the record, which is how a ply caller restates one field and leaves the file's own layout standing for the rest
                target = {}
                for name, entry in self._meta_data.items():
                    target[name] = dict(entry)
                    if meta_data is not None and name in meta_data:
                        target[name].update(meta_data[name])
                for name, entry in (meta_data or {}).items():
                    if name not in self._meta_data:
                        # a field the record never saw defines neither half, so the caller supplies the one that says which columns it is
                        assert (
                            'layout' in entry
                        ), f"a field the record never saw defines neither half, so the caller supplies the layout that says which columns it is: name={name}, entry={entry}, recorded fields={tuple(self._meta_data.keys())}"
                        target[name] = dict(entry)
                stated_columns = set()
                for entry in (meta_data or {}).values():
                    if 'layout' in entry:
                        stated_columns.update(entry['layout'])
                for name, entry in self._meta_data.items():
                    if meta_data is not None and name in meta_data:
                        continue
                    if any(column in stated_columns for column in entry['layout']):
                        # a caller-stated layout CONSUMES the source columns it names, so a column assembled into the field the caller asked for does not also survive under the field the record had assembled it into
                        del target[name]
            for name in list(target.keys()):
                if name not in self._meta_data:
                    continue
                if (
                    not any(column in table for column in target[name]['layout'])
                    and name not in self._fields
                ):
                    # the field was deleted, and save writes no column for one the obj no longer holds
                    del target[name]
            for name, value in self._fields.items():
                # a field the record names is one the record speaks for, so what this loop reaches is the fields that arrived after it
                if name in target or name in self._meta_data:
                    continue
                if not any(name in entry['layout'] for entry in target.values()):
                    # a field assigned after construction takes both halves from itself, while a column another entry's layout already assembles is that field's column rather than a field of its own
                    target[name] = {
                        'layout': (name,),
                        'dtype': CONCEPTUAL_NAME[value.dtype],
                    }
            for name, entry in target.items():
                if all(column in table for column in entry['layout']):
                    dtypes = set(table[column][1] for column in entry['layout'])
                    assert (
                        len(dtypes) == 1
                    ), f"columns that disagree abort rather than being promoted to a dtype that covers them all: name={name}, layout={entry['layout']}, column dtypes={ {column: table[column][1] for column in entry['layout']} }"
                    if 'dtype' not in entry:
                        # a half the override leaves out is the one the record defines
                        entry['dtype'] = next(iter(dtypes))
                elif not any(column in table for column in entry['layout']):
                    assert (
                        name in self._fields
                    ), f"an entry reaching neither the cloud's own columns nor a field it holds names nothing at all, which is a misspelling rather than a field to drop in silence: name={name}, layout={entry['layout']}, source columns={tuple(table.keys())}, fields={tuple(self._fields.keys())}"
                    if meta_data is not None and 'layout' in meta_data.get(name, {}):
                        assert len(entry['layout']) == self._fields[name].shape[1], (
                            "the reverse mapping writes one output column per name, so a count that disagrees leaves the writer with no name for a column: "
                            f"name={name}, layout={entry['layout']}, value.shape={tuple(self._fields[name].shape)}"
                        )
                    if 'dtype' not in entry:
                        entry['dtype'] = (
                            self._meta_data[name]['dtype']
                            if name in self._meta_data
                            else CONCEPTUAL_NAME[self._fields[name].dtype]
                        )
                else:
                    assert 0, (
                        "a layout is either the source columns a field is assembled from or the names its columns are written out under, never a mixture, and a half-matching one is a misspelling rather than either: "
                        f"name={name}, layout={entry['layout']}, source columns={tuple(table.keys())}"
                    )
            # a cloud whose columns assemble into no coordinate field is legal here, a reader building one from a positional source having no coordinates to name yet
            return target

        target = _derive_target_meta_data(table=table)

        def _apply_target_meta_data(
            table: Dict[str, Tuple[torch.Tensor, str]],
            target: Dict[str, Dict[str, Any]],
        ) -> None:
            """Rebuilds every field the target names out of the columns its layout names, on the convention and at the width that entry states.

            Args:
                table: The source columns keyed by column name, each value a pair of the column's values as a torch tensor of shape [N] or [N, C] and the conceptual dtype name that column means.
                target: The target meta data, one entry per field keyed by field name, each entry holding a 'dtype' naming a conceptual dtype and a 'layout' naming a tuple of source column names.

            Returns:
                None.
            """
            fields = {}
            for name, entry in target.items():
                if all(column in table for column in entry['layout']):
                    # a one-dimensional column becomes one column wide and a block that is already two-dimensional keeps the width it has, which is what lets three ply columns and one pcd attribute reach the same [N, 3]
                    value = torch.cat(
                        [
                            (
                                table[column][0]
                                if table[column][0].ndim >= 2
                                else table[column][0].unsqueeze(-1)
                            )
                            for column in entry['layout']
                        ],
                        dim=1,
                    )
                    # the dtype the source held, which the record keeps whatever the target asks the values to become
                    source_dtype = next(
                        iter(set(table[column][1] for column in entry['layout']))
                    )
                else:
                    # the layout is naming this field's output columns rather than selecting the cloud's own, so nothing is regrouped
                    value = self._fields[name]
                    source_dtype = (
                        self._meta_data[name]['dtype']
                        if name in self._meta_data
                        else CONCEPTUAL_NAME[value.dtype]
                    )
                # a stated dtype that is what the columns already MEAN moves nothing, so a record naming one width over a tensor of another leaves that tensor where it is
                if (
                    meta_data is not None
                    and name in meta_data
                    and 'dtype' in meta_data[name]
                    and entry['dtype'] != source_dtype
                ):
                    if name == 'rgb':
                        converted = convert_color_convention(
                            values=value,
                            source_dtype=source_dtype,
                            target_dtype=entry['dtype'],
                        )
                        # the mapping hands back double precision, and a colour is held at the storage its own convention names
                        converted = converted.to(TORCH_DTYPE[entry['dtype']])
                        assert bool(
                            (
                                convert_color_convention(
                                    values=converted,
                                    source_dtype=entry['dtype'],
                                    target_dtype=source_dtype,
                                )
                                == value
                            ).all()
                        ), f"tolerating a rounded colour belongs to a display converting its own copy, never to the cloud these colours are the record of: name={name}, source_dtype={source_dtype}, target dtype={entry['dtype']}"
                        value = converted
                        # the colours sit on the target's range now, so that is the convention they MEAN and the one the record has to name for the next reader to read them by
                        source_dtype = entry['dtype']
                    else:
                        # a narrowing the target cannot hold exactly aborts inside the cast, a caller wanting one narrowing its own values before handing them in
                        value = cast_lossless(
                            values=value, dtype=TORCH_DTYPE[entry['dtype']]
                        )
                fields[name] = value
                if meta_data is not None and name in meta_data:
                    # the layout half a caller states is what the field now IS, while the dtype half stays what its columns held, so the override moves the mapping's loaded side and never the provenance
                    self._meta_data[name] = {
                        'layout': entry['layout'],
                        'dtype': source_dtype,
                    }
            self._fields = fields

        _apply_target_meta_data(table=table, target=target)

        for name, value in self._fields.items():
            self._assert_field_name_valid(name=name)
            # both slots are assigned by now, so a colour is bounded by the convention the record it was just made to match names
            self._validate_field(name=name, value=value)

        # the dtype half a caller states never reaches the record, so the target is handed back for the writer that has to write the file at it
        return target

    @property
    def device(self) -> torch.device:
        """Hands back the one device every field of this point cloud sits on.

        Args:
            None.

        Returns:
            The torch device every field sits on.
        """
        return self._device

    @property
    def num_points(self) -> int:
        """Hands back the number of points every field carries.

        Args:
            None.

        Returns:
            The number of points, as an int.
        """
        return self._length

    @property
    def meta_data(self) -> Dict[str, Dict[str, Any]]:
        """Hands back the meta data this point cloud carries: one entry per field, each holding the conceptual dtype that field's source held and the source columns it was assembled from.

        Args:
            None.

        Returns:
            The recorded meta data, one entry per field keyed by field name, each entry holding a 'dtype' naming a conceptual dtype and a 'layout' naming a tuple of source column names.
        """
        return self._meta_data

    def field_names(self) -> Tuple[str, ...]:
        """Hands back every field name this point cloud carries, coordinates first because they entered first.

        Args:
            None.

        Returns:
            The field names, as a tuple of str in the order the fields entered.
        """
        names = tuple(self._fields.keys())
        return names

    def __len__(self) -> int:
        """Serves the point count to len(), so a point cloud measures as its number of points.

        Args:
            None.

        Returns:
            The number of points, as an int.
        """
        return self._length

    def __getattr__(self, name: str) -> torch.Tensor:
        """Serves any field as an attribute under its own name, coordinates included, for a name ordinary attribute lookup did not find.

        Args:
            name: The attribute name ordinary lookup did not find, as a str.

        Returns:
            The field stored under name, as a torch tensor of shape [N] or [N, C].
        """
        assert all(
            slot in self.__dict__
            for slot in ('_fields', '_meta_data', '_length', '_device')
        ), f"a point cloud serves its fields only once its own slots are assigned: name={name}, assigned slots={tuple(self.__dict__.keys())}"
        if name in self._fields:
            return self._fields[name]
        # the name is no field this point cloud carries
        raise AttributeError(name)

    def __setattr__(self, name: str, value: object) -> None:
        """Routes an assignment to the private slot for an underscore name, and to a validated field otherwise, leaving the meta data exactly as it stands.

        Args:
            name: The attribute name being assigned, as a str; an underscore-prefixed one names one of this class's own slots and any other names a field.
            value: The value being assigned; one of the dicts, the length or the device this class stores about itself on the slot branch, and a torch tensor of shape [N] or [N, C] on the field branch.

        Returns:
            None.
        """
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        self._assert_field_name_valid(name=name)
        self._validate_field(name=name, value=value)
        # the meta data is left exactly as it stands, and what the field now means follows from the two
        self._fields[name] = value

    def __delattr__(self, name: str) -> None:
        """Removes a field, leaving the meta data exactly as it stands.

        Args:
            name: The field name to remove, as a str.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert (
                name != 'xyz'
            ), f"a point cloud without coordinates is not one: name={name}, fields={tuple(self._fields.keys())}"
            assert (
                name in self._fields
            ), f"only a field this point cloud carries can be removed: name={name}, fields={tuple(self._fields.keys())}"

        _validate_inputs()

        # the meta data goes on naming the departed field, and save simply writes no column for one the obj no longer holds
        del self._fields[name]

    def __getstate__(self) -> dict:
        """Hands the four private slots to pickle, so a point cloud and its meta data survive a round trip across a process boundary.

        Args:
            None.

        Returns:
            The pickled state, as a dict holding the four private slots '_fields', '_meta_data', '_length' and '_device' under their own names.
        """
        state = {
            '_fields': self._fields,
            '_meta_data': self._meta_data,
            '_length': self._length,
            '_device': self._device,
        }
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores the four private slots from a pickled state dict.

        Args:
            state: The pickled state, as a dict holding the four private slots '_fields', '_meta_data', '_length' and '_device' under their own names.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert isinstance(
                state, dict
            ), f"a pickled point cloud state arrives as a dict: type(state)={type(state)}"
            assert all(
                slot in state
                for slot in ('_fields', '_meta_data', '_length', '_device')
            ), f"a payload written before the meta data existed carries no such slot and is refused here, to be regenerated rather than accepted through a shim: state keys={tuple(state.keys())}"

        _validate_inputs()

        self._fields = state['_fields']
        self._meta_data = state['_meta_data']
        self._length = state['_length']
        self._device = state['_device']

    def _validate_field(self, name: str, value: torch.Tensor) -> None:
        """Checks one field's tensor-ness, rank, length and device, then the extra rules the names xyz and rgb carry.

        Args:
            name: The field name the value is being stored under, as a str.
            value: The field's values, as a torch tensor of shape [N] or [N, C] on this point cloud's device.

        Returns:
            None.
        """
        assert isinstance(
            value, torch.Tensor
        ), f"a field is a torch tensor: name={name}, type(value)={type(value)}"
        assert (
            value.ndim >= 1
        ), f"a field carries at least one dimension: name={name}, value.shape={tuple(value.shape)}"
        assert (
            value.shape[0] >= 1
        ), f"a field carries at least one point: name={name}, value.shape={tuple(value.shape)}"
        assert (
            value.shape[0] == self._length
        ), f"every field carries the same number of points: name={name}, value.shape={tuple(value.shape)}, num_points={self._length}"
        assert (
            value.device == self._device
        ), f"every field sits on this point cloud's device: name={name}, value.device={value.device}, device={self._device}"
        if name == 'xyz':
            self.validate_xyz_tensor(value)
        elif name == 'rgb':
            # the record is what says an int32 tensor holds a uint16 colour, and a colour under a name the record never saw is exact in its own tensor
            color_dtype = (
                self._meta_data[name]['dtype']
                if name in self._meta_data
                else CONCEPTUAL_NAME[value.dtype]
            )
            self.validate_rgb_tensor(value, color_dtype)

    @staticmethod
    def validate_xyz_tensor(xyz: torch.Tensor) -> None:
        """Checks coordinates are an [N, 3] floating point tensor of any width, free of NaN and Inf.

        Args:
            xyz: The coordinates, as a torch tensor of shape [N, 3] carrying any floating point dtype.

        Returns:
            None.
        """
        assert isinstance(
            xyz, torch.Tensor
        ), f"coordinates are a torch tensor: type(xyz)={type(xyz)}"
        assert (
            xyz.ndim == 2
        ), f"coordinates are two-dimensional: xyz.shape={tuple(xyz.shape)}"
        assert (
            xyz.shape[1] == 3
        ), f"coordinates carry three columns: xyz.shape={tuple(xyz.shape)}"
        assert (
            xyz.is_floating_point()
        ), f"coordinates are a floating point tensor: xyz.dtype={xyz.dtype}"
        assert not bool(
            torch.isnan(xyz).any()
        ), f"xyz tensor contains NaN: number of NaN entries={int(torch.isnan(xyz).sum())}"
        assert not bool(
            torch.isinf(xyz).any()
        ), f"xyz tensor contains Inf: number of Inf entries={int(torch.isinf(xyz).sum())}"

    @staticmethod
    def validate_rgb_tensor(rgb: torch.Tensor, current_dtype: str) -> None:
        """Checks colors are an [N, 3] tensor whose values sit inside the range of the colour convention the dtype they MEAN names, which is not the range of the tensor parking them.

        Args:
            rgb: The colours, as a torch tensor of shape [N, 3] carrying the values of the convention current_dtype names, whatever dtype the tensor itself parks them in.
            current_dtype: The conceptual dtype name whose colour convention the values are read on, as a string keying COLOR_RANGE.

        Returns:
            None.
        """
        assert isinstance(
            rgb, torch.Tensor
        ), f"colours are a torch tensor: type(rgb)={type(rgb)}"
        assert (
            rgb.ndim == 2
        ), f"colours are two-dimensional: rgb.shape={tuple(rgb.shape)}"
        assert (
            rgb.shape[1] == 3
        ), f"colours carry three columns: rgb.shape={tuple(rgb.shape)}"
        assert not bool(
            torch.isnan(rgb).any()
        ), f"colours carry no NaN: number of NaN entries={int(torch.isnan(rgb).sum())}"
        assert not bool(
            torch.isinf(rgb).any()
        ), f"colours carry no Inf: number of Inf entries={int(torch.isinf(rgb).sum())}"
        # a bool or int32 colour names no convention at all, and its absence from the table is what refuses it
        assert (
            current_dtype in COLOR_RANGE
        ), f"a dtype naming no colour convention bounds no colour: current_dtype={current_dtype}, dtypes naming a convention={sorted(COLOR_RANGE.keys())}"
        # the conventions are told apart by dtype, never by inspecting the values, so a uint16 colour parked in an int32 tensor is bounded by uint16's range rather than int32's
        assert bool(
            (
                (rgb >= COLOR_RANGE[current_dtype][0])
                & (rgb <= COLOR_RANGE[current_dtype][1])
            ).all()
        ), f"colours sit inside the bounds of the convention their dtype names: current_dtype={current_dtype}, bounds={COLOR_RANGE[current_dtype]}, rgb.min()={float(rgb.min())}, rgb.max()={float(rgb.max())}"

    def _assert_field_name_valid(self, name: str) -> None:
        """Checks a field name is a str, is not underscore-prefixed, and collides with none of the reserved attribute names.

        Args:
            name: The field name to check, as a str.

        Returns:
            None.
        """
        assert isinstance(name, str), f"a field name is a str: type(name)={type(name)}"
        assert not name.startswith(
            '_'
        ), f"a field name does not start with an underscore, which names this class's own slots: name={name}"
        # a field under a name the class already binds would be written and then never readable
        assert name not in (
            'device',
            'num_points',
            'meta_data',
            'field_names',
            'apply_meta_data',
            'validate_xyz_tensor',
            'validate_rgb_tensor',
        ), f"a field under a name the class already binds would be written and then never readable: name={name}"
