# Point Cloud Data Structure Tests Structure

## Tests implementation structure

`tests/data/structures/three_d/test_point_cloud.py`

```text
test_point_cloud.py
├── import copy
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.select import Select
├── from utils.input_checks.check_point_cloud import check_point_cloud_segmentation
├── def test_point_cloud_keys_and_access
│   ├── # A point cloud built from coordinates alone reports the point count and the device those coordinates carry.
│   ├── impls xyz = a [4, 3] float32 random tensor
│   ├── calls PointCloud(xyz=xyz)
│   ├── impls assert num_points is 4
│   └── impls assert device is the device of xyz
├── def test_setitem_validation
│   ├── # Assigning a field of the wrong length is refused, and one of the right length lands.
│   ├── calls PointCloud(xyz=a [5, 3] float32 random tensor)
│   ├── with pytest.raises(AssertionError)
│   │   └── impls assigns a [4, 2] tensor to the feat attribute
│   ├── impls assigns a [5, 2] tensor to the feat attribute
│   └── impls assert feat reads back
├── def test_missing_field_access
│   ├── # Reading a field the point cloud does not carry raises AttributeError.
│   ├── calls PointCloud(xyz=a [3, 3] float32 random tensor)
│   └── with pytest.raises(AttributeError)
│       └── impls reads the feat attribute
├── def test_point_cloud_requires_xyz
│   ├── # A cloud whose columns assemble into no coordinate field is legal here, a positional reader building exactly one, and load_point_cloud's own door is where a loaded cloud without coordinates is refused.
│   ├── calls PointCloud(data=a dict carrying feat alone)
│   └── impls assert it carries feat and no xyz
├── def test_point_cloud_rejects_nan_xyz
│   ├── # Coordinates carrying NaN are refused, under the message that names the NaN.
│   └── with pytest.raises(AssertionError, match="xyz tensor contains NaN")
│       └── calls PointCloud(xyz=a [1, 3] tensor whose first entry is NaN)
├── def test_point_cloud_rejects_inf_xyz
│   ├── # Coordinates carrying Inf are refused on the same terms as NaN, which the coordinate validator checks separately.
│   └── with pytest.raises(AssertionError, match="xyz tensor contains Inf")
│       └── calls PointCloud(xyz=a [1, 3] tensor whose first entry is Inf)
├── def test_rgb_carrying_nan_or_inf_is_refused
│   ├── # The colour validator rejects both on its own, so a float colour field cannot smuggle either past the range check.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls PointCloud(data={'xyz': a [1, 3] float32 tensor, 'rgb': a [1, 3] float32 tensor carrying NaN})
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [1, 3] float32 tensor, 'rgb': a [1, 3] float32 tensor carrying Inf})
├── def test_coordinates_given_twice_are_refused
│   ├── # Coordinates arrive through one route or the other, never both, so a field dict carrying xyz beside the xyz argument aborts.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(xyz=a [4, 3] float32 tensor, data={'xyz': a [4, 3] float32 tensor})
├── def test_a_field_on_another_device_is_refused
│   ├── # Every field of one cloud sits on one device, so a field arriving on another is refused rather than silently moved.
│   ├── impls skipped unless torch.cuda.is_available()
│   ├── calls PointCloud(xyz=a [4, 3] float32 tensor on the cpu)
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a [4, 2] cuda tensor to the feat attribute
├── def test_a_field_that_is_not_a_tensor_is_refused
│   ├── # Every field is a tensor, so a None or a bare list is refused at the door rather than carried and skipped later by whatever consumes it.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': None})
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': a plain list of four floats})
├── def test_an_underscore_field_name_is_refused
│   ├── # An underscore name is the class's own private slot namespace, so a field may not take one.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, '_secret': a [4] float32 tensor})
├── def test_point_cloud_length_mismatch_on_assignment
│   ├── # A field assigned onto an existing point cloud must carry as many points as the coordinates do.
│   ├── calls PointCloud(xyz=a [5, 3] float32 random tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a [4, 2] tensor to the feat attribute
├── def test_reserved_attribute_assignment_rejected
│   ├── # A field may not be assigned under one of the reserved attribute names.
│   ├── calls PointCloud(xyz=a [3, 3] float32 random tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a tensor to the device attribute
├── def test_non_string_keys_rejected
│   ├── # A field dict keyed by anything but a str is refused.
│   ├── impls xyz = a [4, 3] float32 random tensor
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data=a dict keyed by 'xyz' and by the integer 1)
├── def test_point_cloud_segmentation_validation
│   ├── # Matching segmentation logits and labels pass the checker through untouched.
│   ├── impls logits = a [6, 4] float32 random tensor
│   ├── impls labels = a [6] int64 tensor of class ids
│   ├── calls check_point_cloud_segmentation(y_pred=logits, y_true=labels)
│   └── impls assert the checker hands both tensors back as the very objects it was given
├── def test_rgb_is_admitted_at_any_integer_width
│   ├── # An integer color field is admitted whatever its width, because ply stores colors as u1 while las stores them as uint16.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'rgb': a [4, 3] uint8 tensor})
│   ├── impls assert rgb reads back as uint8
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'rgb': a [4, 3] uint16 numpy array})
│   └── impls assert the meta data of rgb holds 'uint16' while its tensor is torch.int32  # a width naming no colour convention names no colour either, so what "any integer width" reaches is every width COLOR_RANGE bounds
├── def test_a_float_rgb_outside_zero_to_one_is_refused
│   ├── # A float color field declares the 0 to 1 convention by its dtype, so a value outside that range is refused.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [2, 3] float32 tensor, 'rgb': a [2, 3] float32 tensor holding 0 to 255 values})
├── def test_xyz_of_the_wrong_shape_is_refused
│   ├── # The coordinate validator's rank and width guards each stand on their own input, so neither can be what holds the other up.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls PointCloud(xyz=a [4] float32 tensor)
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(xyz=a [4, 4] float32 tensor)
├── def test_a_zero_dimensional_field_is_refused
│   ├── # Every field is indexed by point, so a scalar carrying no point axis at all is refused before its length is ever compared.
│   ├── calls PointCloud(xyz=a [4, 3] float32 tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a zero-dimensional tensor to the feat attribute
├── def test_a_field_carrying_no_points_is_refused
│   ├── # A cloud of no points is not a cloud, and this guard is exercised on its own rather than through a reader that would also trip the length comparison.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(xyz=a [0, 3] float32 tensor)
├── def test_rgb_of_the_wrong_shape_is_refused
│   ├── # The colour validator's rank and width guards each stand on their own input, so neither can be what holds the other up.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'rgb': a [4] uint8 tensor})
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'rgb': a [4, 4] uint8 tensor})
├── def test_the_two_color_conventions_are_told_apart_by_dtype_alone
│   ├── # An integer color field holding only zeros and ones is still the integer convention, because the dtype decides and the values are never inspected.
│   ├── calls PointCloud(data={'xyz': a [2, 3] float32 tensor, 'rgb': a [2, 3] uint8 tensor holding only 0 and 1})
│   └── impls assert rgb reads back as uint8
├── def test_a_colour_is_bounded_by_the_range_it_MEANS_not_the_one_it_is_parked_in
│   ├── # torch has no uint16, so a uint16 colour sits in an int32 tensor; the range enforced is uint16's own, or the guard would pass anything int32 can hold and let a colour walk out of its convention.
│   ├── calls PointCloud(data={'xyz': a [2, 3] float32 numpy array, 'rgb': a [2, 3] uint16 numpy array})
│   ├── impls assert the stored rgb tensor is torch.int32 and its field dtype is 'uint16'
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a [2, 3] int32 tensor holding a value above 65535 to the rgb attribute
├── def test_xyz_is_admitted_at_any_floating_point_width
│   ├── # Coordinates are any floating point dtype, so an f8 source is not narrowed to float32.
│   ├── calls PointCloud(xyz=a [4, 3] float64 tensor)
│   └── impls assert xyz reads back as float64
├── def test_indices_no_longer_have_to_be_int64
│   ├── # PointCloud treats a field named indices as an ordinary field, and nothing downstream reimposes the dtype the design retires.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'indices': a [4] int32 tensor})
│   ├── impls assert indices read back as int32
│   ├── calls Select([0, 2])(the cloud it built)
│   └── impls assert the selected cloud's indices are still int32  # a selection indexes that field rather than consuming it, so its dtype survives
├── def test_a_field_under_a_name_the_class_binds_is_refused
│   ├── # A field written under a public attribute name would land in the field dict and then never be readable, since ordinary lookup finds the class attribute first.
│   ├── calls PointCloud(xyz=a [3, 3] float32 tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a tensor to the validate_rgb_tensor attribute
├── def test_conceptual_dtype_is_a_reserved_field_name
│   ├── # conceptual_dtype is the method naming the dtype a field means, so a field under that name would land in the field dict and then never be readable.
│   ├── calls PointCloud(xyz=a [3, 3] float32 tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a tensor to the conceptual_dtype attribute
├── def test_a_deleted_field_leaves_the_meta_data_naming_it
│   ├── # The meta data is what construction saw, so deleting a field takes the field and leaves the meta data exactly as it was, and save then writes no column for a field the obj no longer holds.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': a [4, 2] float32 tensor})
│   ├── impls deletes the feat attribute
│   ├── impls assert field_names() no longer carries feat
│   └── impls assert the meta data entry for column feat is still {'dtype': 'float32', 'field': 'feat'}
├── def test_coordinates_cannot_be_deleted
│   ├── # Every other field may leave, but a point cloud without coordinates is not one, so the coordinate field is the one deletion refused.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': a [4, 2] float32 tensor})
│   └── with pytest.raises(AssertionError)
│       └── impls deletes the xyz attribute
├── def test_indexing_takes_the_named_rows_of_every_field
│   ├── # Indexing a cloud by an int64 index tensor hands back a new cloud carrying those rows of every field in the order the indices name, and leaves the original whole.
│   ├── calls PointCloud(data={'xyz': a [5, 3] float32 tensor, 'feat': a [5, 2] float32 tensor})
│   ├── impls selected = pc indexed by the int64 tensor [4, 0] on pc's device
│   ├── impls assert selected is a PointCloud of two points
│   ├── impls assert field_names() of selected is ('xyz', 'feat')
│   ├── impls assert its xyz is rows 4 and 0 of the original   # impls-node-one-step:skip
│   ├── impls assert its feat is rows 4 and 0 of the original  # impls-node-one-step:skip
│   └── impls assert pc still carries five points
├── def test_indexing_refuses_an_empty_index_tensor
│   ├── # A cloud of no points is not one, so an index tensor naming no point is refused at the door.
│   ├── calls PointCloud(xyz=a [3, 3] float32 tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls indexes pc by an empty int64 tensor on its device
├── def test_indexing_refuses_an_index_out_of_range
│   ├── # An index naming a point the cloud does not have is refused at the door, whether it runs past the last point or would wrap round from the far end.
│   ├── calls PointCloud(xyz=a [3, 3] float32 tensor)
│   ├── with pytest.raises(AssertionError)
│   │   └── impls indexes pc by the int64 tensor [0, 3] on its device
│   └── with pytest.raises(AssertionError)
│       └── impls indexes pc by the int64 tensor [-1] on its device
├── def test_indexing_refuses_a_non_int64_index_tensor
│   ├── # A cloud is indexed by an int64 tensor alone, so an int32 one is refused rather than cast.
│   ├── calls PointCloud(xyz=a [3, 3] float32 tensor)
│   └── with pytest.raises(AssertionError)
│       └── impls indexes pc by the int32 tensor [0, 2] on its device
├── def test_a_copy_holds_a_field_dict_of_its_own
│   ├── # A copy assigns and deletes fields in a dict of its own, so changing the copy's fields leaves the original's exactly as they were.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': a [4, 2] float32 tensor})
│   ├── calls copy.copy(pc)
│   ├── impls copied = the copy it made
│   ├── impls assigns a [4, 1] float32 tensor to the label attribute of copied
│   ├── impls deletes the feat attribute of copied
│   ├── impls assert field_names() of copied is ('xyz', 'label')
│   └── impls assert field_names() of pc is still ('xyz', 'feat')
└── def test_point_cloud_segmentation_validation_errors
    ├── # Segmentation logits and labels of different lengths are refused.
    ├── impls logits = a [5, 3] float32 random tensor
    ├── impls labels = a [4] int64 tensor of class ids
    └── with pytest.raises(AssertionError)
        └── calls check_point_cloud_segmentation(y_pred=logits, y_true=labels)
```

`tests/data/structures/three_d/point_cloud/test_point_cloud_meta_data.py`

```text
test_point_cloud_meta_data.py
├── import copy
├── import pickle
├── import numpy as np
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── def test_the_construction_meta_data_records_the_dtype_the_source_held
│   ├── # A field entering as a numpy uint16 array notes uint16, whatever torch had to store it as.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls assert the meta data entry for intensity holds 'uint16'
│   └── impls assert the stored intensity tensor is torch.int32
├── def test_a_dtype_override_moves_the_values_onto_the_dtype_it_states
│   ├── # The meta data a constructor is handed overrides what the source held, so a stated dtype moves the values while the record goes on holding the source's own.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': a [4] float32 tensor}, meta_data={'feat': {'dtype': 'float64', 'layout': ('a',)}})
│   ├── impls assert the stored feat tensor is torch.float64
│   ├── impls assert the meta data entry for column feat is {'dtype': 'float32', 'field': 'feat'}
│   ├── calls pc.apply_meta_data()  # an application stating nothing hands back the target the cloud stands on
│   └── impls assert what it returned maps feat to {'dtype': 'float64', 'layout': ('a',)}  # a layout naming columns the data does not hold names the field's own block afresh, one name per column it carries
├── def test_the_constructor_casts_nothing_a_caller_did_not_hand_it
│   ├── # A caller wanting another dtype hands the field over in it, so the one cast here is the crossing into torch and a float64 field stays float64.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'feat': a [4] float64 numpy array})
│   ├── impls assert the stored feat tensor is torch.float64
│   └── impls assert the meta data entry for feat holds 'float64'
├── def test_a_meta_data_entry_may_state_its_dtype_alone
│   ├── # An override states one half or both, so an entry stating a dtype alone keeps the columns its field already stands for.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array}, meta_data={'xyz': {'dtype': 'float32'}})
│   ├── calls pc.apply_meta_data()
│   └── impls assert what it returned maps xyz to {'dtype': 'float32', 'layout': ('xyz',)}
├── def test_a_bool_field_is_carried_as_its_own_kind
│   ├── # bool is a kind of its own in the dtype universe, not a narrow integer, so a mask field enters as bool and is noted as bool rather than being widened to one.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 torch tensor, 'visible': a [4] bool torch tensor})
│   ├── impls assert visible is stored as torch.bool
│   └── impls assert the meta data entry for visible holds 'bool'
├── def test_a_bool_colour_is_refused
│   ├── # A colour spans a range its dtype declares, and two values are not a range to span, so bool is refused at the colour door while staying a legal field dtype elsewhere.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 torch tensor, 'rgb': a [4, 3] bool torch tensor})
├── def test_a_bfloat16_field_is_stored_without_a_numpy_detour
│   ├── # bfloat16 is torch's alone, so a field entering as one is cast in torch rather than through a numpy dtype that does not exist.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 torch tensor, 'feat': a [4] bfloat16 torch tensor})
│   ├── impls assert feat is stored as torch.bfloat16
│   └── impls assert the meta data entry for feat holds 'bfloat16'
├── def test_a_meta_entry_stating_an_empty_layout_is_refused
│   ├── # A field assembled from no columns at all is not a field, so the door refuses the entry rather than reading past it.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array}, meta_data={'xyz': {'dtype': 'float32', 'layout': ()}})  # the dtype half is legal on its own, so the empty layout is the only thing left to refuse
├── def test_a_meta_entry_naming_a_half_the_design_has_no_slot_for_is_refused
│   ├── # An entry carries a dtype half, a layout half or both, so a misspelled key aborts rather than being read past in silence.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array}, meta_data={'xyz': {'dytpe': 'float32'}})
├── def test_a_meta_entry_stating_neither_half_is_refused
│   ├── # An entry that names no dtype and no layout asks for nothing, so it is refused rather than silently ignored downstream.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array}, meta_data={'xyz': {}})
├── def test_a_layout_override_names_the_output_columns_of_an_in_memory_field
│   ├── # A stated layout naming columns the data does not hold names an in-memory field's own block afresh, which is where the columns that field is written back under are declared.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array}, meta_data={'xyz': {'dtype': 'float32', 'layout': ('x', 'y', 'z')}})
│   ├── impls assert xyz is still one [4, 3] field
│   ├── impls assert the meta data entry for column xyz is {'dtype': 'float32', 'field': 'xyz'}
│   ├── calls pc.apply_meta_data()
│   └── impls assert what it returned maps xyz onto ('x', 'y', 'z')
├── def test_a_layout_naming_a_block_afresh_with_another_column_count_is_refused
│   ├── # A layout naming a block afresh names one column per column the block carries, so two names over a three-column block are refused rather than leaving a column unnamed.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array}, meta_data={'xyz': {'layout': ('x', 'y')}})
├── def test_x_y_and_z_columns_stay_three_fields_until_a_layout_assembles_them
│   ├── # A field keeps the name its column arrived under, so coordinates handed in as x, y and z columns become one xyz field only where a stated layout assembles them.
│   ├── impls columns = {'x': a [4] float32 numpy array, 'y': a [4] float32 numpy array, 'z': a [4] float32 numpy array}
│   ├── calls PointCloud(data=columns)
│   ├── impls assert field_names() of the cloud it built is ('x', 'y', 'z')
│   ├── calls PointCloud(data=columns, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   └── impls assert field_names() of the cloud it built is ('xyz',), its xyz being [4, 3]
├── def test_a_dtype_override_that_would_change_a_value_is_refused
│   ├── # A stated dtype moves the values rather than only naming them, so a narrowing a value does not survive aborts instead of rounding it.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float64 numpy array carrying a value float32 cannot hold exactly}, meta_data={'xyz': {'dtype': 'float32'}})
├── def test_an_in_memory_field_gets_the_identity_mapping
│   ├── # A field handed in under a name and no layout gets that name standing for its whole block, rather than nothing.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'feat': a [4] float32 numpy array})
│   ├── impls assert the meta data is {'xyz': {'dtype': 'float32', 'field': 'xyz'}, 'feat': {'dtype': 'float32', 'field': 'feat'}}
│   ├── calls pc.apply_meta_data()
│   └── impls assert what it returned maps xyz onto ('xyz',) and feat onto ('feat',)
├── def test_a_float128_field_is_stored_in_the_widest_float_torch_carries
│   ├── # float128 is ruled in or out by whether its values survive the narrowing, not by its name, so a field whose values fit float64 enters and keeps the meta data entry for what its source held.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'feat': a [4] float128 numpy array whose values are exactly representable in float64})
│   ├── impls assert feat is stored as torch.float64
│   └── impls assert the meta data entry for feat holds 'float128'
├── def test_a_float128_field_whose_values_need_its_width_is_refused
│   ├── # The same dtype is ruled out by the same test when the values do not survive, which is what makes the rule about values rather than names.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'feat': a [4] float128 numpy array carrying a value float64 cannot hold exactly})
├── def test_a_complex_field_is_stored_through_its_component_float
│   ├── # A complex dtype is compared through the float its parts are, so a complex256 field whose parts fit float64 enters as complex128.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'feat': a [4] complex256 numpy array whose parts are exactly representable in float64})
│   ├── impls assert feat is stored as torch.complex128
│   └── impls assert the meta data entry for feat holds 'complex256'
├── def test_complex_coordinates_are_refused
│   ├── # Coordinates are a floating point field whatever the dtype universe admits elsewhere, so a complex block is refused at the coordinate door rather than the dtype one.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(xyz=a [4, 3] complex128 numpy array)
├── def test_a_uint64_source_is_refused
│   ├── # uint64 is unsupported as a source dtype whatever values it carries, so no representability test is reached at all.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'ids': a [4] uint64 numpy array holding small values})
├── def test_an_override_naming_another_dtype_does_not_rescue_a_uint64_source
│   ├── # The refusal reads the value's own width, so no override over it makes a uint64 array admissible.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'ids': a [4] uint64 numpy array}, meta_data={'ids': {'dtype': 'int64', 'layout': ('ids',)}})
├── def test_an_override_stating_uint64_is_refused
│   ├── # uint64 is refused as a dtype an override states the way it is as a source dtype, whatever values it would move.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'ids': a [4] int64 numpy array holding small non-negative values}, meta_data={'ids': {'dtype': 'uint64'}})
├── def test_an_overwrite_leaves_the_meta_data_untouched
│   ├── # A user may modify a field, but the meta data stays exactly what entered with it.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls assigns a [4] int64 tensor to the intensity attribute
│   ├── impls assert the meta data entry for intensity still holds 'uint16'
│   └── impls assert the stored intensity tensor is torch.int64
├── def test_a_parked_field_means_the_dtype_its_source_held
│   ├── # torch has no uint16, so a uint16 column is parked in an int32 tensor, and conceptual_dtype is what names the dtype those int32 values mean.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls assert the stored intensity tensor is torch.int32
│   └── impls assert pc.conceptual_dtype('intensity') is 'uint16'
├── def test_an_overwritten_field_means_the_dtype_its_new_tensor_carries
│   ├── # An int64 tensor is exact in its own dtype, so an intensity overwritten with one means int64 while the record goes on holding the uint16 the source held.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls assigns a [4] int64 tensor to the intensity attribute
│   ├── impls assert pc.conceptual_dtype('intensity') is 'int64'
│   └── impls assert the meta data entry for column intensity still holds 'uint16'
├── def test_replacing_rgb_with_a_clone_preserves_its_colour_convention
│   ├── # A clone sits at the int32 storage a uint16 colour is parked in, so it goes on meaning uint16 and the convention a display reads off it is unchanged.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'rgb': a [4, 3] uint16 numpy array})
│   ├── impls assigns a clone of the stored rgb tensor to the rgb attribute
│   ├── impls assert pc.conceptual_dtype('rgb') is still 'uint16', its tensor being torch.int32
│   └── impls assert the meta data entry for rgb still holds 'uint16'
├── def test_a_float_rgb_outside_zero_to_one_is_refused_on_a_later_assignment
│   ├── # The colour range is enforced on every assignment, not only at the door, so a field cannot be walked out of its own convention after it enters.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'rgb': a [4, 3] float32 numpy array of values in 0 to 1})
│   └── with pytest.raises(AssertionError)
│       └── impls assigns a [4, 3] float32 tensor holding 0 to 255 values to the rgb attribute
├── def test_a_float_colour_on_the_255_grid_converts_to_uint8_exactly
│   ├── # A stated rgb dtype converts the convention losslessly, and a float32 colour at k/255 comes back exactly on float32's own grid, so it enters as the uint8 k.
│   ├── impls k = a [4, 3] uint8 numpy array
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'rgb': k divided by 255 in float32}, meta_data={'rgb': {'dtype': 'uint8'}})
│   ├── impls assert the stored rgb tensor is torch.uint8 and equals k
│   ├── impls assert pc.conceptual_dtype('rgb') is 'uint8'
│   └── impls assert the meta data entry for column rgb still holds 'float32'
├── def test_a_colour_conversion_that_would_round_is_refused
│   ├── # 0.5 lies between two uint8 levels, so its conversion rounds and comes back as another value, and a lossy conversion aborts rather than being made.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'rgb': a [4, 3] float32 numpy array holding 0.5}, meta_data={'rgb': {'dtype': 'uint8'}})
├── def test_a_field_assigned_after_construction_is_named_by_no_meta_data
│   ├── # The meta data is created once and never again, so a field arriving by attribute assignment sits outside it and carries only the dtype it means now.
│   ├── calls PointCloud(xyz=a [4, 3] float32 tensor)
│   ├── impls assigns a [4, 2] float64 tensor to the feat attribute
│   ├── impls assert the meta data names feat nowhere
│   ├── impls assert its tensor is float64, which is what a field outside the meta data means
│   └── with pytest.raises(KeyError)
│       └── impls reads the meta data entry for feat  # the record is a plain mapping, so a name it does not hold is missing rather than refused
├── def test_a_field_assigned_after_the_last_application_means_its_own_dtype
│   ├── # A field assigned after the last application stands outside the target, so the dtype it means is the one its own tensor carries.
│   ├── calls PointCloud(xyz=a [4, 3] float32 tensor)
│   ├── impls assigns a [4, 2] int32 tensor to the feat attribute
│   └── impls assert pc.conceptual_dtype('feat') is 'int32'
├── def test_conceptual_dtype_refuses_a_name_the_cloud_no_longer_carries
│   ├── # A dtype is meant by a field the cloud holds, so a deleted field's name is refused though the target it was applied on still names it.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls deletes the intensity attribute
│   └── with pytest.raises(AssertionError)
│       └── calls pc.conceptual_dtype('intensity')
├── def test_mutating_the_meta_data_handed_back_changes_nothing_in_the_cloud
│   ├── # The record is never mutable, so the meta data property hands back a copy a reader may write through without reaching the cloud.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls record = pc.meta_data
│   ├── impls record['intensity']['dtype'] = 'int8'  # a write into a nested entry, which a shallow copy would carry into the cloud
│   ├── impls record pops its 'xyz' entry
│   └── impls assert pc.meta_data is still {'xyz': {'dtype': 'float32', 'field': 'xyz'}, 'intensity': {'dtype': 'uint16', 'field': 'intensity'}}
├── def test_a_cloud_derived_from_another_carries_its_record_and_its_target
│   ├── # A cloud indexed down or copied is not a source, so it carries its original's record and target whole, which is what says its int32 tensor holds a uint16 field.
│   ├── calls PointCloud(data={'x': a [4] float32 numpy array, 'y': a [4] float32 numpy array, 'z': a [4] float32 numpy array, 'intensity': a [4] uint16 numpy array}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── calls copy.copy(pc)
│   ├── impls copied = the copy it made
│   ├── impls selected = pc indexed by the int64 tensor [0, 2] on pc's device
│   ├── calls pc.apply_meta_data()
│   ├── impls target = what it returned
│   └── for each derived in (copied, selected)
│       ├── impls assert derived.meta_data equals pc.meta_data
│       ├── impls assert derived.conceptual_dtype('intensity') is 'uint16', its tensor being torch.int32
│       ├── calls derived.apply_meta_data()
│       └── impls assert what it returned equals target, xyz mapped onto ('x', 'y', 'z')
├── def test_a_lone_dtype_for_a_field_the_cloud_does_not_hold_is_refused
│   ├── # A dtype half alone keeps the columns its field already stands for, so one stated for a field the cloud does not hold names nothing and is refused as a misspelling.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'feat': a [4] float32 tensor}, meta_data={'departed': {'dtype': 'float32'}})
├── @pytest.mark.parametrize def test_a_payload_lacking_any_of_the_five_slots_is_refused(missing_slot)  # over each of '_fields', '_meta_data', '_target', '_length' and '_device'
│   ├── # A pickle written before any one slot existed would restore a cloud missing part of what it is, so it is refused, to be regenerated rather than restored incomplete.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float32 numpy array, 'intensity': a [4] uint16 numpy array})
│   ├── impls state = the state pc hands pickle, with its missing_slot entry removed
│   └── with pytest.raises(AssertionError)
│       └── impls restores a PointCloud from that state
├── def test_a_point_cloud_survives_a_pickle_round_trip
│   ├── # The record and the target travel through pickle with the fields, so a cloud crossing a process boundary is the same cloud on the other side.
│   ├── calls PointCloud(data={'x': a [4] float32 numpy array, 'y': a [4] float32 numpy array, 'z': a [4] float32 numpy array, 'intensity': a [4] uint16 numpy array}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── impls restored = the cloud round-tripped through pickle.loads(pickle.dumps(pc))
│   ├── impls assert field_names() on restored is ('xyz', 'intensity')
│   ├── impls assert every field of restored equals pc's under the same name, at the same dtype
│   ├── impls assert restored.meta_data equals pc.meta_data, its entry for column intensity being {'dtype': 'uint16', 'field': 'intensity'}
│   ├── impls assert restored.conceptual_dtype('intensity') is 'uint16'
│   ├── calls pc.apply_meta_data()
│   ├── impls target = what it returned
│   ├── calls restored.apply_meta_data()
│   └── impls assert what it returned equals target, xyz mapped onto ('x', 'y', 'z')
├── def test_coordinates_lead_whatever_order_the_fields_arrive_in
│   ├── # Coordinates-first is insertion order now, so the constructor enters them first rather than trusting the caller's dict order.
│   ├── calls PointCloud(data={'feat': a [4] float32 tensor, 'xyz': a [4, 3] float32 tensor})
│   └── impls assert field_names() is ('xyz', 'feat')
├── def test_the_target_lists_coordinates_first_whatever_order_their_columns_arrived_in
│   ├── # Coordinates lead the fields wherever the columns they are assembled from sat among the source's columns, so the target and field_names() both open on xyz.
│   ├── calls PointCloud(data={'intensity': a [4] uint16 numpy array, 'x': a [4] float32 numpy array, 'y': a [4] float32 numpy array, 'z': a [4] float32 numpy array}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── impls assert field_names() is ('xyz', 'intensity')
│   ├── calls pc.apply_meta_data()
│   └── impls assert the first key of what it returned is 'xyz'
├── def test_a_layout_repeating_a_column_is_refused
│   ├── # A column assembled into one field twice is not a layout, and the door refuses the entry rather than reading past it.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'xyz': a [4, 3] float32 tensor}, meta_data={'xyz': {'dtype': 'float32', 'layout': ('a', 'a')}})
├── def test_two_layouts_claiming_one_column_are_refused
│   ├── # A column is assembled into one field and written under one name, so two stated layouts both naming it are refused rather than one silently taking it.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'x': a [4] float32 tensor, 'y': a [4] float32 tensor, 'z': a [4] float32 tensor}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}, 'ground_plane': {'layout': ('x', 'y')}})
├── def test_a_layout_mixing_held_columns_with_unknown_names_is_refused
│   ├── # A layout either assembles columns the cloud holds or names one field's own columns afresh, so one half-matching the held columns is a misspelling and is refused.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data={'x': a [4] float32 tensor, 'y': a [4] float32 tensor, 'z': a [4] float32 tensor}, meta_data={'xyz': {'layout': ('x', 'y', 'w')}})
├── def test_applying_the_same_meta_data_twice_changes_nothing
│   ├── # Every load applies once inside the construction and again on the way out, and every save applies once more, so a second application that moved anything would move it on every ordinary path.
│   ├── impls meta_data = {'xyz': {'layout': ('x', 'y', 'z')}, 'intensity': {'dtype': 'int32'}}
│   ├── calls PointCloud(data={'x': a [4] float32 numpy array, 'y': a [4] float32 numpy array, 'z': a [4] float32 numpy array, 'intensity': a [4] uint16 numpy array}, meta_data=meta_data)
│   ├── calls pc.apply_meta_data()
│   ├── impls target = what it returned
│   ├── impls record = pc.meta_data
│   ├── impls fields = a clone of every field pc holds, under its own name
│   ├── calls pc.apply_meta_data(meta_data=meta_data)
│   ├── impls assert what it returned equals target, xyz still mapped onto ('x', 'y', 'z')
│   ├── impls assert pc.meta_data equals record
│   └── impls assert every field pc holds equals its entry in fields, at the same dtype
├── def test_a_field_assembled_from_columns_can_be_split_back_into_them
│   ├── # A later application names the source columns back out of the field an earlier one assembled, which is what makes the same columns reachable however they were last grouped.
│   ├── calls PointCloud(data={'x': a [4] float32 tensor, 'y': a [4] float32 tensor, 'z': a [4] float32 tensor}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── calls pc.apply_meta_data(meta_data={'ground_plane': {'layout': ('x', 'y')}, 'z': {'layout': ('z',)}})
│   ├── impls target = what it returned
│   ├── impls assert ground_plane is two columns wide and z is its own field  # the coordinates are three columns wherever they are named xyz, so a regrouping onto two carries its own name
│   ├── impls assert target maps ground_plane onto ('x', 'y') and z onto ('z',)
│   └── impls assert the meta data entry for each of columns x, y and z still names xyz as its field
├── def test_a_column_a_partial_claim_leaves_out_survives_under_its_own_name
│   ├── # A stated layout takes the place of the field whose columns it claims, and a column of that field no stated layout names stands as a field of its own name.
│   ├── calls PointCloud(data={'x': a [4] float32 tensor, 'y': a [4] float32 tensor, 'z': a [4] float32 tensor}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── calls pc.apply_meta_data(meta_data={'ground_plane': {'layout': ('x', 'y')}})
│   ├── impls assert field_names() is ('ground_plane', 'z')
│   └── impls assert what it returned maps z onto ('z',)
├── def test_a_field_named_like_another_fields_column_is_refused_at_the_next_application
│   ├── # A field named like a column another field is assembled from would overwrite that column in the next regrouping, so the next application refuses the pair.
│   ├── calls PointCloud(data={'x': a [4] float32 tensor, 'y': a [4] float32 tensor, 'z': a [4] float32 tensor}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── impls assigns a [4] float32 tensor to the x attribute
│   └── with pytest.raises(AssertionError)
│       └── calls pc.apply_meta_data()
├── def test_a_layout_override_enters_the_record_while_the_dtype_override_does_not
│   ├── # The mapping's loaded side is what a caller asked for, and its dtype half stays what the source held whatever width the values were moved to.
│   ├── impls columns = each of '0', '1' and '2' mapped to a [4] float64 tensor holding values float32 holds exactly
│   ├── calls PointCloud(data=columns, meta_data={'xyz': {'dtype': 'float32', 'layout': ('0', '1', '2')}})
│   ├── impls assert the xyz tensor is torch.float32
│   └── impls assert the meta data entry for each of columns '0', '1' and '2' is {'dtype': 'float64', 'field': 'xyz'}
├── def test_applying_meta_data_hands_back_the_target_it_applied
│   ├── # The dtype half a caller states never reaches the record, so the target is handed back for the writer that has to write the file at it.
│   ├── calls PointCloud(data={'intensity': a [4] uint16 numpy array holding values below 256, 'xyz': a [4, 3] float32 numpy array})
│   ├── calls pc.apply_meta_data(meta_data={'intensity': {'dtype': 'uint8'}})
│   ├── impls target = what it returned
│   ├── impls assert target maps intensity to {'dtype': 'uint8', 'layout': ('intensity',)}
│   ├── impls assert target maps xyz to {'dtype': 'float32', 'layout': ('xyz',)}  # a field the override leaves out stays on the target it stood on
│   ├── impls assert the stored intensity tensor is torch.uint8
│   └── impls assert the meta data entry for column intensity is still {'dtype': 'uint16', 'field': 'intensity'}
└── def test_a_deleted_field_is_dropped_from_the_target_rather_than_aborting_it
    ├── # The record goes on naming a departed field, and the application that follows a deletion drops it instead of failing to find its columns.
    ├── calls PointCloud(data={'xyz': a [4, 3] float32 tensor, 'intensity': a [4] uint16 numpy array})
    ├── impls the intensity attribute of pc is deleted
    ├── calls pc.apply_meta_data()
    └── impls assert what it returned names xyz alone
```

`tests/data/structures/three_d/point_cloud/test_select_random_select.py`

```text
test_select_random_select.py
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.random_select import RandomSelect
├── from data.structures.three_d.point_cloud.select import Select
├── def test_a_selection_carries_the_meta_data_across
│   ├── # The record and the target travel with the fields, so a selected cloud still knows what each column's source held and what each field means.
│   ├── calls PointCloud(data={'xyz': a [5, 3] float32 numpy array, 'intensity': a [5] uint16 numpy array}, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── calls Select(indices=[0, 3])
│   ├── impls out = the selection applied to that point cloud
│   ├── impls assert out.meta_data equals the point cloud's, its entry for column intensity being {'dtype': 'uint16', 'field': 'intensity'}
│   ├── impls assert out.conceptual_dtype('intensity') is 'uint16', its tensor being torch.int32
│   ├── calls out.apply_meta_data()
│   └── impls assert what it returned maps xyz onto ('x', 'y', 'z')
├── def test_the_indices_a_selection_makes_are_named_by_no_meta_data
│   ├── # A selection hands its cloud the record and the target whole, so the indices field it adds stands behind no source column and means the dtype its own tensor carries.
│   ├── calls PointCloud(xyz=a [5, 3] float32 tensor)
│   ├── calls Select(indices=[1, 2])
│   ├── impls out = the selection applied to that point cloud
│   ├── impls assert out.meta_data equals the point cloud's, naming column xyz alone
│   ├── impls assert its indices are the one-dimensional int64 tensor [1, 2] of shape [2]  # the [K] tensor a consumer indexes a sibling tensor with
│   └── impls assert out.conceptual_dtype('indices') is 'int64'
├── def test_pointcloud_initialization
│   ├── # A point cloud built from a field dict reports its point count, its field names in coordinates-first order, and its coordinates.
│   ├── impls xyz = a [4, 3] float32 tensor
│   ├── impls feat = a [4, 2] float32 tensor
│   ├── calls PointCloud(data={'xyz': xyz, 'feat': feat})
│   ├── impls assert num_points is 4
│   ├── impls assert field_names() is ('xyz', 'feat')
│   └── impls assert xyz reads back
├── def test_select_takes_a_plain_index_list
│   ├── # Selecting by a plain index list carries every field down and adds the taken indices as a field.
│   ├── calls PointCloud(data={'xyz': a [5, 3] tensor, 'feat': a [5, 1] tensor})
│   ├── calls Select(indices=[0, 3])
│   ├── impls out = the selection applied to that point cloud
│   ├── impls assert out is a PointCloud
│   ├── impls assert its xyz is rows 0 and 3 of the original   # impls-node-one-step:skip
│   ├── impls assert its feat is rows 0 and 3 of the original  # impls-node-one-step:skip
│   └── impls assert its indices are the int64 tensor [0, 3]
├── @pytest.mark.parametrize def test_select_pointcloud(xyz_values, feat_values, indices, expected_xyz_indices, expected_feat_indices)  # over one coordinates / features / indices case
│   ├── # Each parametrized selection takes exactly the named rows of every field and adds the indices it took as a field.
│   ├── calls PointCloud(data={'xyz': xyz_values, 'feat': feat_values})
│   ├── calls Select(indices=indices)
│   ├── impls out = the selection applied to that point cloud
│   ├── impls assert its xyz is xyz_values at expected_xyz_indices
│   ├── impls assert its feat is feat_values at expected_feat_indices
│   └── impls assert its indices are the int64 tensor of indices
└── @pytest.mark.parametrize def test_random_select_pointcloud(count, seed, num_points)  # over the (3, 0, 10) and (5, 1, 20) count / seed / size triples
    ├── # A seeded random selection of a fixed count hands back that many points and an int64 index field of the same length.
    ├── calls PointCloud(xyz=a [num_points, 3] random tensor)
    ├── calls RandomSelect(count=count)
    ├── impls out = the selection applied to that point cloud, under seed
    ├── impls assert out is a PointCloud
    ├── impls assert its num_points is the smaller of count and num_points
    ├── impls assert its indices are int64
    └── impls assert its indices carry one entry per selected point
```

`tests/utils/point_cloud_ops/test_select.py`

```text
test_select.py
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.select import Select
├── def test_select_basic_list
│   ├── # An index list takes those rows of coordinates, colors and a classification field alike, and adds them as the indices field.
│   ├── calls PointCloud(data={'xyz': a [5, 3] float64 tensor, 'rgb': a [5, 3] float64 tensor of values in 0 to 1, 'classification': a [5] int64 tensor})
│   ├── calls Select([0, 2, 4])
│   ├── impls result = the selection applied to that point cloud
│   ├── impls assert its xyz is rows 0, 2 and 4 of the original             # impls-node-one-step:skip
│   ├── impls assert its rgb is rows 0, 2 and 4 of the original             # impls-node-one-step:skip
│   ├── impls assert its classification is rows 0, 2 and 4 of the original  # impls-node-one-step:skip
│   └── impls assert its indices are the int64 tensor [0, 2, 4]
├── def test_select_basic_tensor
│   ├── # An int64 index tensor on the point cloud's device selects exactly as an index list does.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor, 'rgb': a [3, 3] float64 tensor of values in 0 to 1})
│   ├── impls indices_tensor = the int64 tensor [1, 2] on the point cloud's device
│   ├── calls Select(indices_tensor)
│   ├── impls result = the selection applied to that point cloud
│   ├── impls assert its xyz is rows 1 and 2 of the original  # impls-node-one-step:skip
│   └── impls assert its indices are indices_tensor
├── def test_select_empty_indices
│   ├── # Selecting no points at all is refused, because a point cloud carries at least one point.
│   ├── calls PointCloud(data={'xyz': a [2, 3] float64 tensor, 'rgb': a [2, 3] float64 tensor of values in 0 to 1, 'classification': a [2] int64 tensor})
│   ├── calls Select([])
│   └── with pytest.raises(AssertionError)
│       └── impls applies the selection to that point cloud
├── def test_select_single_point
│   ├── # A one-entry selection hands back a one-point cloud whose color field keeps its trailing three columns.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor, 'rgb': a [3, 3] float64 tensor of values in 0 to 1})
│   ├── calls Select([1])
│   ├── impls result = the selection applied to that point cloud
│   ├── impls assert its xyz is row 1 of the original
│   ├── impls assert its rgb is [1, 3]
│   └── impls assert its indices are the int64 tensor [1]
├── def test_select_out_of_order
│   ├── # The selected rows come back in the order the indices name.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float64 tensor})
│   ├── calls Select([3, 0, 2])
│   ├── impls result = the selection applied to that point cloud
│   ├── impls assert its xyz is rows 3, 0 and 2 of the original in that order  # impls-node-one-step:skip
│   └── impls assert its indices are the int64 tensor [3, 0, 2]
├── def test_select_refuses_an_index_past_the_last_point
│   ├── # An index naming a point the cloud does not have is refused at the door rather than reaching torch and coming back as an opaque indexing error.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor})
│   ├── calls Select([0, 5])
│   └── with pytest.raises(AssertionError)
│       └── impls applies the selection to that point cloud
├── def test_select_refuses_a_negative_index
│   ├── # A negative index would silently select from the far end, so it is refused rather than wrapping.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor})
│   ├── calls Select([-1])
│   └── with pytest.raises(AssertionError)
│       └── impls applies the selection to that point cloud
├── def test_select_refuses_a_non_int64_index_tensor
│   ├── # The int64 requirement is asserted on the indices the selection was constructed with, not only on an indices field the cloud already carried.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor})
│   ├── calls Select(the int32 tensor [0, 2])
│   └── with pytest.raises(AssertionError)
│       └── impls applies the selection to that point cloud
├── def test_select_refuses_an_index_tensor_on_another_device
│   ├── # A selection materializes a list onto the cloud's device but never moves a tensor it was handed, so a mismatch aborts instead of transferring silently.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor on cpu})
│   ├── calls Select(an int64 tensor on cuda)
│   └── with pytest.raises(AssertionError)
│       └── impls applies the selection to that point cloud
└── def test_select_duplicate_indices
    ├── # A repeated index takes the same row again, so the selection may be longer than the point cloud it came from.
    ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor})
    ├── calls Select([1, 1, 2, 1])
    ├── impls result = the selection applied to that point cloud
    ├── impls assert its xyz is rows 1, 1, 2 and 1 of the original in that order  # impls-node-one-step:skip
    └── impls assert its indices are the int64 tensor [1, 1, 2, 1]
```

`tests/utils/point_cloud_ops/test_random_select.py`

```text
test_random_select.py
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.random_select import RandomSelect
├── def test_random_select_percentage_basic
│   ├── # A percentage selection keeps that fraction of the points and carries the color field and the index field down with it.
│   ├── calls PointCloud(data={'xyz': a [4, 3] float64 tensor, 'rgb': a [4, 3] float64 tensor of values in 0 to 1})
│   ├── calls RandomSelect(percentage=0.5)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   ├── impls assert its num_points is half of four
│   ├── impls assert its rgb carries one row per selected point
│   ├── impls assert its indices carry one entry per selected point
│   ├── impls assert its indices are int64
│   └── impls assert the meta data entry for rgb is the one the source cloud carried  # a selection indexes fields down, it does not re-derive what they came from
├── def test_random_select_count_basic
│   ├── # A count selection of fewer points than the cloud carries hands back exactly that many.
│   ├── calls PointCloud(xyz=a [5, 3] float64 tensor)
│   ├── calls RandomSelect(count=3)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   ├── impls assert its num_points is 3
│   └── impls assert its indices carry three entries
├── def test_random_select_deterministic_with_seed
│   ├── # Two selections under the same seed draw the very same points in the very same order.
│   ├── calls PointCloud(xyz=a [4, 3] float64 tensor)
│   ├── calls RandomSelect(percentage=0.5)
│   ├── impls result1 = the selection applied to that point cloud under seed 42
│   ├── impls result2 = the selection applied again under seed 42
│   ├── impls assert the two xyz agree
│   └── impls assert the two indices agree
├── def test_random_select_count_exceeds_points
│   ├── # A count larger than the cloud is capped at the number of points there are.
│   ├── calls PointCloud(xyz=a [2, 3] float64 tensor)
│   ├── calls RandomSelect(count=5)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   ├── impls assert its num_points is 2
│   └── impls assert its indices carry two entries
├── def test_random_select_takes_a_quarter_of_twenty
│   ├── # A quarter of a twenty-point cloud is five points.
│   ├── calls PointCloud(xyz=a [20, 3] float64 tensor)
│   ├── calls RandomSelect(percentage=0.25)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   └── impls assert its num_points is a quarter of twenty
├── def test_random_select_takes_ten_of_twenty
│   ├── # A count of ten out of a twenty-point cloud is ten points.
│   ├── calls PointCloud(xyz=a [20, 3] float64 tensor)
│   ├── calls RandomSelect(count=10)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   └── impls assert its num_points is 10
├── def test_random_select_takes_exactly_one_sizing_mode
│   ├── # The two modes size the selection differently, so naming both or neither leaves it undefined rather than defaulting to one.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls RandomSelect()
│   └── with pytest.raises(AssertionError)
│       └── calls RandomSelect(percentage=0.5, count=3)
├── def test_random_select_refuses_a_percentage_outside_its_range
│   ├── # A percentage at or below zero selects no points and one above one selects more than there are, so both are refused where the mode is chosen.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls RandomSelect(percentage=0.0)
│   └── with pytest.raises(AssertionError)
│       └── calls RandomSelect(percentage=1.5)
├── def test_random_select_refuses_a_count_that_is_not_positive
│   ├── # A selection of no points is not a point cloud, so the count is refused at construction rather than producing one downstream.
│   └── with pytest.raises(AssertionError)
│       └── calls RandomSelect(count=0)
└── def test_random_select_takes_exactly_one_source_of_randomness
    ├── # A seed and a generator are two ways to fix the same draw, so naming both or neither leaves the draw undefined.
    ├── calls PointCloud(xyz=a [8, 3] float64 tensor)
    ├── calls RandomSelect(count=3)
    ├── with pytest.raises(AssertionError)
    │   └── impls applies the selection with neither a seed nor a generator
    └── with pytest.raises(AssertionError)
        └── impls applies the selection with both a seed and a generator
```
