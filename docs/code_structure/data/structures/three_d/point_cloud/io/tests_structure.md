# Point Cloud IO Tests Structure

## Tests implementation structure

`tests/utils/io/point_clouds/load_point_cloud/test_point_cloud_loading.py`

```text
test_point_cloud_loading.py
├── import numpy as np
├── import pytest
├── import torch
├── from plyfile import PlyData, PlyElement
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud, _load_from_ply, _load_from_txt, _load_from_pth, _load_from_off
├── def temp_dir
│   ├── # Yields a fresh temporary directory so each test writes its own files.
│   ├── # Every test below takes this fixture and joins a name onto it, so a bare filepath in one of them is that join rather than a name from nowhere.
│   └── impls yields the path of a tempfile.TemporaryDirectory
├── def test_ply_xyz_only
│   ├── # A PLY carrying only coordinates loads to a PointCloud whose xyz is [N, 3].
│   ├── calls write_ply(filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the result is a PointCloud
│   ├── impls assert its xyz is [N, 3]
│   └── impls assert its xyz is float32
├── def test_ply_with_rgb
│   ├── # RGB columns arrive as an rgb field in the dtype the file stored them in.
│   ├── calls write_ply(filepath, with_rgb=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert rgb is [N, 3]
│   └── impls assert rgb is still uint8
├── def test_ply_extra_field_keeps_its_own_name
│   ├── # A non-standard PLY column is loaded under the name the file gives it, not renamed to feat.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the loaded fields carry intensity
│   └── impls assert the loaded fields carry no feat
├── def test_a_partial_colour_set_stays_separate_columns
│   ├── # The colour mapping is all three columns or none, so a file carrying red and green alone yields two fields under their own names rather than an rgb the third column is missing from.
│   ├── calls write_ply(filepath, colour_columns=('red', 'green'))
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the loaded fields carry red and green and no rgb
│   ├── impls pc = the cloud it loaded
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['red']['layout'] == ('red',)
├── def test_a_layout_assembles_the_columns_it_names
│   ├── # The layout half chooses which source columns a field is assembled from, which is the control that replaced renaming a column to feat.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'feat': {'layout': ('intensity',)}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert feat is present
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   ├── impls assert target['feat']['layout'] == ('intensity',)
│   └── impls assert intensity is absent  # a caller-stated layout CONSUMES its columns, so the column does not also survive under its own name
├── def test_a_caller_layout_over_the_colour_columns_makes_the_rgb_default_yield
│   ├── # The caller's meta data outranks the default layout the reader applied, so a caller assembling red, green and blue into a field of its own takes them out of rgb while the xyz default still stands.
│   ├── calls write_ply(filepath, with_rgb=True)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'colour': {'layout': ('red', 'green', 'blue')}}, device='cpu')
│   ├── impls assert the loaded fields are xyz and colour
│   └── impls assert colour is a [N, 3] uint8 tensor holding the written colours
├── def test_a_caller_layout_over_the_coordinate_columns_leaves_the_load_without_xyz
│   ├── # The coordinates yield the same way, so a caller assembling x, y and z under another name takes them from the default and the load aborts for lacking xyz.
│   ├── calls write_ply(filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'pos': {'layout': ('x', 'y', 'z')}}, device='cpu')
├── def test_the_record_keys_each_source_column_and_the_target_names_each_field_s_columns
│   ├── # A PLY's default layout assembles its coordinate columns into xyz and its colour columns into rgb, and the record keeps what each of those six source columns held beside the field the reader's construction assembled it into.
│   ├── calls write_ply(filepath, with_rgb=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert pc.meta_data holds x, y and z each as {'dtype': 'float32', 'field': 'xyz'}
│   ├── impls assert pc.meta_data holds red, green and blue each as {'dtype': 'uint8', 'field': 'rgb'}  # the record is written when the reader constructs the cloud, which is where ply's default layout is applied
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   ├── impls assert target['xyz']['layout'] == ('x', 'y', 'z')
│   └── impls assert target['rgb']['layout'] == ('red', 'green', 'blue')
├── def test_a_source_defining_no_layout_is_assembled_by_the_caller
│   ├── # A .pth names nothing about its block, so its columns come back under their own indices and the caller's layout says which of them each field is assembled from.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'label': {'layout': ('3',)}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   ├── impls assert target['xyz']['layout'] == ('0', '1', '2')
│   ├── impls assert target['label']['layout'] == ('3',)
│   └── impls assert xyz is [N, 3] and label is [N, 1]
├── def test_a_caller_may_assemble_the_same_block_differently
│   ├── # The reader divines nothing from the column count, so one file loads two ways and neither is the reader's choice.
│   ├── calls torch.save(a [N, 7] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'rgb': {'layout': ('3', '4', '5')}, 'label': {'layout': ('6',)}}, device='cpu')
│   ├── impls assert rgb is [N, 3]
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3', '4', '5', '6')}}, device='cpu')
│   └── impls assert feat is [N, 4] and rgb is absent
├── def test_a_source_defining_no_layout_refuses_a_load_that_states_none
│   ├── # A source that defines no layout makes the caller's meta data required rather than optional, so a load without it aborts instead of guessing.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_a_layout_naming_a_column_index_the_block_lacks_is_refused
│   ├── # The caller selects among the columns the block actually holds, so an index past its width aborts rather than assembling a field from nothing.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '9')}}, device='cpu')
├── def test_a_layout_repeating_a_column_is_refused_at_load
│   ├── # A caller-stated layout never passes through a meta data entry on the way in either, so its distinctness is checked at this door as well as at save's.
│   ├── calls write_ply(filepath, with_rgb=True)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'feat': {'layout': ('red', 'red')}}, device='cpu')
├── def test_an_empty_layout_is_refused_at_load
│   ├── # A field assembled from no source columns at all is not a field, so an empty tuple aborts rather than producing one.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'feat': {'layout': ()}}, device='cpu')
├── def test_a_layout_naming_a_column_the_file_lacks_is_refused
│   ├── # A layout can only choose among columns the file actually holds, so naming one it does not aborts rather than assembling a field from nothing.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'feat': {'layout': ('nosuchcolumn',)}}, device='cpu')
├── def test_a_file_without_coordinate_columns_is_refused
│   ├── # A PLY's default layout assembles xyz from its x, y and z columns, so a file naming none of them under a caller naming no coordinates of its own leaves the load without xyz.
│   ├── calls write_ply(filepath, without_coordinates=True, extra_field='intensity')  # the intensity column keeps the raw cloud buildable, so the missing coordinates are what the load refuses
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_a_ply_carrying_no_element_is_refused
│   ├── # A PLY's columns are its elements' properties, so a file declaring no element at all has none and is refused where it is read.
│   ├── calls write_ply(filepath, without_elements=True)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_a_pth_holding_something_other_than_a_block_is_refused
│   ├── # A .pth is one block of columns, so a payload that is a dict or a list is refused rather than being indexed as though it were an array.
│   ├── calls torch.save({'not': 'a block'}, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
├── def test_a_pth_holding_a_block_of_one_axis_is_refused
│   ├── # Columns are keyed by index along a second axis, so a block that has none is refused at this door rather than raising an IndexError from inside the split.
│   ├── calls torch.save(a [N] float32 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
├── def test_a_pcd_loaded_without_meta_data_is_refused
│   ├── # A pcd defines no default layout, so its positions attribute stays a field under that name and a load stating no layout for xyz holds no coordinates.
│   ├── calls write_pcd(filepath, with_colors=True)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_a_pcd_whose_stated_layout_names_positions_it_lacks_is_refused
│   ├── # positions is an attribute name like any other, so a caller's layout naming it over a pcd carrying none names a column the file lacks.
│   ├── calls write_pcd(filepath, with_colors=True, without_positions=True)  # the colors attribute keeps the raw cloud buildable, so the stated layout is what the load refuses
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('positions',)}}, device='cpu')
├── def test_a_las_bit_packed_dimension_enters_as_an_ordinary_uint8_field
│   ├── # laspy materializes a bit-packed dimension as uint8, so uint8 is what enters the obj and what the record stores, with no special treatment.
│   ├── calls write_las(filepath, with_classification=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert the return_number field is torch.uint8
│   └── impls assert pc.meta_data['return_number'] == {'dtype': 'uint8', 'field': 'return_number'}
├── def test_a_las_uint16_colour_keeps_its_own_width
│   ├── # las stores colours as uint16, and a reader neither widens nor narrows, so the colour goes on meaning uint16 while torch parks it in int32.
│   ├── calls write_las(filepath, with_rgb=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert the rgb field is torch.int32
│   ├── impls assert pc.conceptual_dtype(name='rgb') == 'uint16'  # the meaning and the storage differ here, which is the divergence a display reads the wrong side of when it reads the tensor
│   ├── impls assert pc.meta_data holds red, green and blue each as {'dtype': 'uint16', 'field': 'rgb'}
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['rgb']['layout'] == ('red', 'green', 'blue')
├── def test_a_las_dimension_the_file_does_not_carry_is_skipped
│   ├── # A point format that has no colour does not name red, green and blue among its dimensions at all, so they become no column rather than an empty one.
│   ├── calls write_las(filepath, with_rgb=False)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the loaded fields carry no rgb
│   └── impls assert the loaded fields carry xyz and return_number, the dimensions this file does hold
├── def test_a_las_coordinate_is_the_scaled_value_not_the_raw_dimension
│   ├── # A las stores its coordinates as integers to be multiplied by the header's scale and shifted by its offset, so the dimension's own values are not points and reading them as points misplaces every one.
│   ├── calls write_las(filepath, scales=(0.001, 0.001, 0.001), offsets=(100.0, 200.0, 300.0))
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert xyz holds the real-world coordinates the file was written with, not the integers the dimensions store
│   ├── impls assert pc.meta_data holds x, y and z each as {'dtype': 'float64', 'field': 'xyz'}  # the coordinate's own dtype is the scaled float, and las's default layout assembles the three into xyz
│   ├── impls assert pc.meta_data holds no X, Y or Z  # the raw int32 dimension is the container's storage of both, the same divergence a uint16 colour has inside an int32 tensor
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['xyz']['layout'] == ('x', 'y', 'z')
├── def test_a_laz_file_loads_as_its_las_counterpart_does
│   ├── # The compressed container changes nothing about the dimensions laspy hands back.
│   ├── calls write_las(filepath, with_rgb=True, compressed=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert the coordinates match the written ones
│   └── impls assert pc.conceptual_dtype(name='rgb') == 'uint16' and pc.meta_data['red']['dtype'] == 'uint16'
├── def test_a_pcd_s_attributes_are_assembled_by_the_layouts_the_caller_states
│   ├── # A pcd's attributes are named source columns under no default layout, so the caller's layouts over those names are what assemble xyz and rgb.
│   ├── calls write_pcd(filepath, with_colors=True)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('positions',)}, 'rgb': {'layout': ('colors',)}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert xyz is [N, 3] and rgb is [N, 3]
│   ├── impls assert pc.meta_data['positions']['field'] == 'positions' and pc.meta_data['colors']['field'] == 'colors'  # the record keeps each attribute under its own name, whatever field a layout later assembles it into
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['xyz']['layout'] == ('positions',) and target['rgb']['layout'] == ('colors',)
├── def test_a_pcd_attribute_beyond_positions_and_colors_keeps_its_own_name
│   ├── # An attribute no stated layout names stays a field under its own Open3D name, beside the fields the caller's layouts assemble.
│   ├── calls write_pcd(filepath, with_colors=True, extra_attribute='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('positions',)}, 'rgb': {'layout': ('colors',)}}, device='cpu')
│   └── impls assert the loaded fields are xyz, rgb and intensity
├── def test_a_pcd_colour_arrives_as_uint8_whatever_the_writer_held
│   ├── # PCD stores colour as bytes, so Open3D scales a float colour by 255 on the way out and the field that comes back means the 0-to-255 convention rather than the 0-to-1 one it was written in.
│   ├── calls write_pcd(filepath, with_colors=True, colors_dtype='float32')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('positions',)}, 'rgb': {'layout': ('colors',)}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert pc.meta_data['colors']['dtype'] == 'uint8'  # the record states the file, not the writer's intent, which is what keeps the convention read off it truthful
│   └── impls assert rgb spans 0 to 255 rather than 0 to 1
├── def test_a_pcd_unsigned_attribute_reaches_the_cloud_at_its_own_width
│   ├── # Open3D carries unsigned widths torch has none of, so the reader's route to numpy is what decides whether a uint16 attribute arrives as uint16 or not at all.
│   ├── calls write_pcd(filepath, with_colors=True, extra_attribute='intensity', extra_attribute_dtype='uint16')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('positions',)}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert pc.meta_data['intensity']['dtype'] == 'uint16' and pc.conceptual_dtype(name='intensity') == 'uint16'
│   └── impls assert the loaded intensity carries the values the file held  # a route through torch raises before reaching this, since torch names no unsigned width above uint8
├── def test_ply_custom_element_name
│   ├── # The one element a PLY carries is the file's to name, so an element called anything at all is the one that is read.
│   ├── calls write_ply(filepath, element_name='points')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz is [N, 3]
├── def test_a_multi_element_ply_is_assembled_from_the_layouts_the_caller_names
│   ├── # A file with more than one element does not define which element's columns a field comes from, so the layout the caller states names them and the columns are addressed element-qualified.
│   ├── calls write_ply(filepath, extra_element=True)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('vertex.x', 'vertex.y', 'vertex.z')}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert xyz is [N, 3]
│   ├── impls assert pc.meta_data['vertex.x'] == {'dtype': 'float32', 'field': 'vertex.x'}
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['xyz']['layout'] == ('vertex.x', 'vertex.y', 'vertex.z')
├── def test_a_multi_element_ply_refuses_a_load_that_states_no_layout
│   ├── # Without the caller naming them there is nothing to choose between two elements' columns, so the load aborts rather than picking one.
│   ├── calls write_ply(filepath, extra_element=True)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_txt_columns_are_assembled_only_by_the_caller
│   ├── # The text reader decides nothing about which columns are coordinates, colours or labels; the layout the caller states does, so the same file loads differently for different callers and no width is divined.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'rgb': {'layout': ('3', '4', '5')}, 'label': {'layout': ('6',)}}, device='cpu')
│   ├── impls assert rgb holds the fourth, fifth and sixth columns
│   └── impls assert label holds the seventh column
├── def test_a_txt_column_no_layout_names_stays_a_field_under_its_index
│   ├── # Every column of a positional source is a field under its index until a stated layout claims it, so the columns beyond the coordinates come back beside xyz.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert the loaded fields are xyz, '3', '4', '5' and '6'
├── def test_txt_columns_the_caller_names_as_a_colour_are_validated_as_one
│   ├── # Naming three float columns rgb is the caller asking for a colour field, so the 0-to-1 rule applies to them and a 0-to-255 file aborts on the caller's own naming rather than on a guess the reader made.
│   ├── calls write_txt(filepath, num_columns=7, color_scale=255)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'rgb': {'layout': ('3', '4', '5')}, 'label': {'layout': ('6',)}}, device='cpu')
├── def test_txt_columns_the_caller_names_as_features_are_not
│   ├── # The same file loads without complaint when the caller names those columns as an ordinary feature field, since no colour convention is claimed for them.
│   ├── calls write_txt(filepath, num_columns=7, color_scale=255)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3', '4', '5', '6')}}, device='cpu')
│   └── impls assert feat is [N, 4]
├── def test_pth_tensor
│   ├── # A saved torch tensor's columns are assembled by the caller, since the file names nothing about them.
│   ├── calls write_pth(filepath, a [N, 4] torch tensor)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cpu')
│   ├── impls assert xyz is [N, 3]
│   └── impls assert feat is [N, 1]
├── def test_pth_ndarray
│   ├── # A saved numpy array is accepted on the same terms as a tensor.
│   ├── calls write_pth(filepath, a [N, 3] np.ndarray)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert xyz is [N, 3]
├── def test_pth_rejects_a_non_array_payload
│   ├── # A .pth holding anything but a tensor or an array is rejected rather than half-read.
│   ├── calls write_pth(filepath, a dict)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
├── def test_a_uint64_source_column_is_refused_at_load
│   ├── # uint64 is unsupported as a source dtype whatever its values, so it is refused on the way in rather than narrowed.
│   ├── calls write_pth(filepath, a [N, 3] uint64 np.ndarray holding small values)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'ids': {'layout': ('a', 'b', 'c')}}, device='cpu')
├── def test_a_uint64_dtype_is_refused_at_load
│   ├── # And naming it as a target dtype does not make it acceptable either, on the load door as on the constructor's.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'intensity': {'dtype': 'uint64'}}, device='cpu')
├── def test_off_xyz_only
│   ├── # An OFF file's vertex block loads as coordinates once the caller's layout names its three positional columns.
│   ├── calls write_off(filepath, four vertices)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert xyz is [4, 3]
├── def test_an_off_loaded_without_meta_data_is_refused
│   ├── # OFF names its vertex columns by position under no default layout, so a load stating no layout for xyz holds no coordinates.
│   ├── calls write_off(filepath, four vertices)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_off_keeps_only_the_leading_three_columns
│   ├── # A vertex line carrying more than three numbers contributes only its coordinates.
│   ├── calls write_off(filepath, vertices of six numbers each)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   ├── impls assert xyz is [N, 3]
│   ├── impls assert xyz holds the leading three of each line
│   └── impls assert the loaded fields are xyz alone  # the reader keeps the leading three columns, which the stated layout consumes whole
├── def test_an_off_coordinate_float32_cannot_hold_is_refused_by_the_reader
│   ├── # float32 is the width this format is read at, so a magnitude beyond float32's range overflows in the parse and the reader itself refuses the file at that width.
│   ├── calls write_off(filepath, four vertices one of whose coordinates is 1e39)
│   └── with pytest.raises(AssertionError)
│       └── calls _load_from_off(filepath=filepath, device='cpu')  # the reader is called directly, since its raw cloud carries no xyz for the coordinate validation to refuse first
├── def test_an_off_ordinary_decimal_loads_at_float32_s_nearest_value
│   ├── # The text lands on float32 directly, so an ordinary decimal such as 0.1 loads as the float32 value nearest to it.
│   ├── calls write_off(filepath, four vertices one of whose coordinates is 0.1)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert xyz is float32 and holds np.float32(0.1) where 0.1 was written
├── def test_off_without_its_header_is_rejected
│   ├── # A file whose first line is not OFF is rejected rather than parsed as vertices.
│   ├── calls write_off(filepath, four vertices, header='INVALID')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')  # the layout is stated so the header is what fails
├── def test_an_off_with_its_counts_glued_to_the_keyword_loads
│   ├── # ModelNet40 writes the keyword and the counts on one line, and refusing that shape would refuse the dataset this design loads .off for.
│   ├── calls write_off(filepath, four vertices, glue_counts_to_header=True)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert xyz is [4, 3] and holds the written coordinates
├── def test_an_off_with_a_comment_before_its_counts_loads
│   ├── # OFF permits a comment line, so the counts are the next line that carries any, rather than the next line whatever it holds.
│   ├── calls write_off(filepath, four vertices, comment='made by something')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert xyz is [4, 3] and holds the written coordinates
├── def test_load_from_ply_returns_the_file_s_columns_assembled_by_ply_s_default_layout
│   ├── # The PLY reader owns ply's default layout, so the cloud it builds already holds xyz and rgb, and every column no default names stays a field under its own name.
│   ├── calls write_ply(filepath, with_rgb=True, extra_field='intensity')
│   ├── calls _load_from_ply(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it built
│   ├── impls assert its fields are xyz, rgb and intensity
│   ├── impls assert each field carries the conceptual dtype the file stored its columns in
│   └── impls assert pc.meta_data holds all seven columns, each with the dtype the file stored it in, and names 'xyz' for x, y and z, 'rgb' for red, green and blue and 'intensity' for intensity
├── def test_load_from_txt_returns_its_columns_under_their_indices
│   ├── # The text reader names its columns by position and nothing else, so a seven-column file hands back seven float64 fields and no field names at all.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls _load_from_txt(filepath=filepath, device='cpu')
│   ├── impls assert its fields are '0' through '6'
│   ├── impls assert every field is float64
│   └── impls assert its fields carry no xyz  # the file defines no coordinate columns, which is what makes the caller's meta data required rather than optional
├── def test_load_from_txt_divines_no_fields_from_the_column_count
│   ├── # Seven columns named xyz, rgb and feat was the reader's own invention, and the width now decides nothing, which is what the dataset's stated meta data replaced.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls _load_from_txt(filepath=filepath, device='cpu')
│   └── impls assert its fields carry no xyz, no rgb and no feat
├── def test_load_from_txt_reads_columns_however_they_are_spaced
│   ├── # Point cloud text is written aligned as often as it is written with single spaces, so the reader splits on runs of whitespace rather than on one named space character.
│   ├── calls write_txt(filepath, num_columns=3, spacing='aligned')
│   ├── calls _load_from_txt(filepath=filepath, device='cpu')
│   ├── impls assert it returned three fields of the written values
│   ├── calls write_txt(filepath, num_columns=3, spacing='single')
│   ├── calls _load_from_txt(filepath=filepath, device='cpu')
│   └── impls assert it returned the same three fields  # one delimiter reads both files, where naming a single space reads only the second
├── def test_a_txt_of_one_row_still_has_columns_to_key
│   ├── # numpy drops the column axis for a file holding one row, so a single-point cloud would arrive with scalars where the columns keyed by index should be.
│   ├── calls write_txt(filepath, num_points=1, num_columns=3)
│   ├── calls _load_from_txt(filepath=filepath, device='cpu')
│   ├── impls assert its fields are '0', '1' and '2'
│   └── impls assert each of those fields holds exactly one entry
├── def test_load_from_pth_returns_a_raw_cloud
│   ├── # The .pth reader builds a cloud whose fields are the block's columns under their own indices, its coordinates unnamed until the caller's meta data names them.
│   ├── calls write_pth(filepath, a [N, 4] torch tensor)
│   ├── calls _load_from_pth(filepath=filepath, device='cpu')
│   ├── impls assert it is a PointCloud
│   └── impls assert its fields are '0' through '3' and carry no xyz
├── def test_load_from_off_returns_a_raw_cloud
│   ├── # OFF names its vertex columns by position, so the reader builds a cloud whose fields are those three columns under their indices, its coordinates unnamed until the caller's meta data names them.
│   ├── calls write_off(filepath, four vertices)
│   ├── calls _load_from_off(filepath=filepath, device='cpu')
│   ├── impls pc = the raw cloud it built
│   ├── impls assert its fields are '0', '1' and '2', each a [4, 1] float32 tensor, and carry no xyz
│   └── impls assert pc.meta_data['0'] == {'dtype': 'float32', 'field': '0'}
├── def test_missing_file_is_rejected
│   ├── # A path naming no file is rejected before any reader is chosen.
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath='nonexistent.ply', device='cpu')
├── def test_unsupported_extension_is_rejected
│   ├── # An extension no reader owns is rejected, and the file's contents are never opened.
│   ├── impls an existing file whose extension is .xyz
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_uppercase_extension_is_rejected
│   ├── # An extension differing from a supported one only in case names no reader, so it is rejected.
│   ├── impls filepath = a path under temp_dir ending in .PLY
│   ├── calls write_ply(filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def write_ply(filepath, num_points=8, with_rgb=False, extra_field=None, element_name='vertex', extra_element=False, without_coordinates=False, without_elements=False, colour_columns=None)
│   ├── # Writes a PLY carrying xyz plus whichever optional columns the caller asks for, a second element when one is asked for, and no coordinate columns at all when the caller wants that case.
│   ├── impls columns = the x, y and z fields as float32, or no columns at all when without_coordinates is set  # impls-node-one-step:skip — names the three fields
│   ├── if with_rgb
│   │   └── impls columns gains red, green and blue as uint8  # impls-node-one-step:skip — names the three fields
│   ├── if extra_field is not None
│   │   └── impls columns gains extra_field as float32
│   ├── calls PlyElement.describe(the rows, element_name)
│   ├── impls elements = the described element, followed by a second described element when extra_element is set
│   └── calls PlyData.write(filepath)
├── def write_txt(filepath, num_points=8, num_columns=3, spacing='single', color_scale=1)
│   ├── # Writes a point cloud behind the two header lines the reader skips, spaced the way the caller asks so one delimiter can be shown to read both shapes.
│   ├── impls two header lines
│   ├── impls the values of columns three, four and five scaled to span 0 to color_scale  # a caller naming those three as rgb decides which convention they are read under, and the reader itself names none
│   ├── if spacing == 'aligned'
│   │   └── impls num_points rows of num_columns floats padded to a fixed column width, which puts runs of spaces between them
│   └── else
│       └── impls num_points rows of num_columns floats separated by one space each
├── def write_pth(filepath, array)
│   ├── # Saves one array as the single block a .pth point cloud holds.
│   └── calls torch.save(array, filepath)
├── def write_off(filepath, vertices, header='OFF', glue_counts_to_header=False, comment=None)
│   ├── # Writes an OFF file in any of the header shapes the format is found in, the keyword being a parameter so a malformed one can be written too.
│   ├── if glue_counts_to_header
│   │   └── impls one line carrying the keyword with the three counts run onto it
│   ├── else
│   │   ├── impls header line
│   │   ├── if comment is not None
│   │   │   └── impls a line opening with a hash and carrying comment
│   │   └── impls the vertex, face and edge counts  # impls-node-one-step:skip — names the three counts
│   └── impls one line per vertex
├── def write_las(filepath, num_points=8, with_rgb=False, with_classification=False, compressed=False, scales=(0.01, 0.01, 0.01), offsets=(0.0, 0.0, 0.0))
│   ├── # Writes a LAS or LAZ file carrying xyz plus, on request, uint16 colour dimensions and the bit-packed return_number laspy materializes as uint8.
│   ├── calls laspy.LasData(a header of the point format that carries the requested dimensions, with scales and offsets set to the caller's)  # a caller pinning these is how the raw dimensions are made to differ visibly from the coordinates they encode
│   ├── impls las_data = the file it built
│   ├── impls the x, y and z dimensions of las_data  # impls-node-one-step:skip — names the three dimensions, as the real-world coordinates laspy converts down into the raw ones
│   ├── if with_rgb
│   │   └── impls the red, green and blue dimensions as mid-range uint16 values
│   ├── if with_classification
│   │   └── impls the bit-packed return_number dimension
│   └── calls las_data.write(filepath)
└── def write_pcd(filepath, num_points=8, with_colors=False, colors_dtype='float32', extra_attribute=None, extra_attribute_dtype='float32', without_positions=False)
    ├── # Writes a PCD through Open3D's tensor IO, so its attributes come back as the whole named blocks the reader hands over under their own names.
    ├── calls o3d.t.geometry.PointCloud(a positions tensor of num_points rows, or no positions attribute when without_positions is set)
    ├── if with_colors
    │   └── impls the colors attribute of that point cloud, in colors_dtype and over the range that dtype names
    ├── if extra_attribute is not None
    │   └── impls a one-column attribute of that point cloud under the name extra_attribute, in extra_attribute_dtype
    └── calls o3d.t.io.write_point_cloud(filepath, the point cloud)
```

`tests/utils/io/point_clouds/load_point_cloud/test_point_cloud_operations.py`

```text
test_point_cloud_operations.py
├── import numpy as np
├── import pytest
├── import torch
├── from plyfile import PlyData, PlyElement
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── def temp_dir
│   ├── # Yields a fresh temporary directory so each test writes its own files.
│   ├── # Every test below takes this fixture and joins a name onto it, so a bare filepath in one of them is that join rather than a name from nowhere.
│   └── impls yields the path of a tempfile.TemporaryDirectory
├── def test_values_survive_the_load
│   ├── # The coordinates that come back are the ones that were saved.
│   ├── calls torch.save(a known [N, 3] tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert xyz equals the saved tensor
├── def test_placed_on_the_requested_device
│   ├── # Every field lands on the device the caller named.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cpu')
│   └── for each name in the result's field names
│       └── impls assert that field's device type is cpu
├── def test_placed_on_cuda_when_asked
│   ├── # The same placement holds for a cuda device, where one is available.
│   ├── impls skipped unless torch.cuda.is_available()
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cuda')
│   └── impls assert xyz's device type is cuda
├── def test_a_dtype_reaches_one_field_at_a_time
│   ├── # The dtype half reaches one field by name, leaving every other field at the dtype its source held.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'dtype': 'float64', 'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert xyz is float64
│   ├── impls assert feat is still float32
│   └── impls assert pc.meta_data holds '0', '1' and '2' each with the dtype 'float32'  # the record keeps what the source columns held
├── def test_a_dtype_reaches_a_field_the_coordinates_are_not
│   ├── # The retired dtype argument could only cast the coordinates; the dtype half reaches any field by name.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'dtype': 'float64', 'layout': ('3',)}}, device='cpu')
│   ├── impls assert feat is float64
│   └── impls assert xyz is still float32
├── def test_a_positional_column_no_layout_names_stays_a_field_under_its_index
│   ├── # A column no stated layout claims enters as the field of its own name, which for a .pth is its index, so every column the block held reaches the cloud.
│   ├── calls torch.save(a [N, 7] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert the loaded fields are xyz, '3', '4', '5' and '6'
├── def test_a_bfloat16_target_is_reached_from_a_float32_source
│   ├── # The dtype a caller states may name the one dtype only torch carries, and the cast runs in the system that has it.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'intensity': {'dtype': 'bfloat16'}}, device='cpu')
│   └── impls assert intensity is stored as torch.bfloat16
├── def test_a_stated_colour_dtype_moves_the_values_onto_the_convention_it_names
│   ├── # A colour's dtype IS its convention, so stating one converts the values onto that range and the colour then means the convention they are on, while the record keeps the one the file held.
│   ├── calls write_ply(filepath, with_rgb=True, colour_dtype='uint16', colour_values='multiples of 257')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'rgb': {'dtype': 'uint8'}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert the stored rgb tensor is torch.uint8 holding the 0-to-255 counterparts of what the file held
│   ├── impls assert pc.conceptual_dtype(name='rgb') == 'uint8'
│   └── impls assert pc.meta_data holds red, green and blue each with the dtype 'uint16'
├── def test_a_stated_colour_dtype_that_would_round_the_values_is_refused
│   ├── # Construction and load refuse a lossy conversion exactly as save does, so a colour that cannot come back is never quietly rounded on the way in.
│   ├── calls write_ply(filepath, with_rgb=True, colour_dtype='uint16', colour_values='1')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'rgb': {'dtype': 'uint8'}}, device='cpu')
├── def test_a_stated_dtype_that_would_narrow_a_value_away_is_refused
│   ├── # Every cast these modules make is lossless, so narrowing is the caller's to do on its own values before handing them in.
│   ├── calls torch.save(a [N, 4] float64 tensor whose coordinates need float64, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'dtype': 'float32', 'layout': ('0', '1', '2')}}, device='cpu')
├── def test_a_dtype_alone_is_enough_where_the_source_defines_the_layout
│   ├── # A ply names its own columns, so a caller who only wants a different dtype states only that and the file's layout stands.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'intensity': {'dtype': 'float64'}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert intensity is float64
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['xyz']['layout'] == ('x', 'y', 'z')
├── def test_a_dtype_only_xyz_entry_rides_on_the_ply_default_layout
│   ├── # A caller's entry is written over the default half by half, so a dtype alone for xyz keeps the x, y and z columns the ply default assembles it from.
│   ├── calls write_ply(filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'dtype': 'float64'}}, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert xyz is a [N, 3] float64 tensor holding the written coordinates
│   ├── impls assert pc.meta_data['x']['dtype'] == 'float32'  # the record keeps what the column held
│   ├── calls pc.apply_meta_data()
│   ├── impls target = the target it handed back
│   └── impls assert target['xyz']['layout'] == ('x', 'y', 'z')
├── def test_a_meta_entry_naming_no_such_field_is_refused_at_load
│   ├── # A name the layout never produces aborts rather than leaving the source dtype silently in force, and the layout it does produce is stated so the misspelling is what fails.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}, 'nosuchfield': {'dtype': 'float64'}}, device='cpu')
├── def test_a_seg_filename_no_longer_casts_the_feature_column
│   ├── # No filename decides a dtype any more: a basename carrying _seg keeps its feature column exactly as stored.
│   ├── calls torch.save(a [N, 4] float32 tensor, a filepath containing _seg)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cpu')
│   └── impls assert feat is still float32
├── def test_feature_column_is_left_alone_without_the_seg_marker
│   ├── # The same file without _seg in its basename keeps its feature column's dtype.
│   ├── calls torch.save(a [N, 4] tensor, a filepath without _seg)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cpu')
│   └── impls assert feat is float32
├── def test_a_u2_column_is_stored_as_int32_and_notes_uint16
│   ├── # torch carries no uint16, so the field is stored in the narrowest torch dtype that holds it while the record keeps what the column held and the field goes on meaning it.
│   ├── calls PlyElement.describe(rows carrying a u2 label column, 'vertex')
│   ├── calls PlyData.write(filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert the label field is torch.int32
│   ├── impls assert pc.meta_data['label'] == {'dtype': 'uint16', 'field': 'label'}
│   └── impls assert pc.conceptual_dtype(name='label') == 'uint16'
├── def test_a_u4_column_is_stored_as_int64_and_notes_uint32
│   ├── # The same patch one width up: torch carries no uint32 either, and the record still keeps the column's own dtype.
│   ├── calls PlyElement.describe(rows carrying a u4 id column, 'vertex')
│   ├── calls PlyData.write(filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls pc = the cloud it loaded
│   ├── impls assert the id field is torch.int64
│   ├── impls assert pc.meta_data['id'] == {'dtype': 'uint32', 'field': 'id'}
│   └── impls assert pc.conceptual_dtype(name='id') == 'uint32'
├── def test_columns_disagreeing_in_dtype_under_no_stated_dtype_are_refused
│   ├── # The columns a layout merges must hold one dtype once the target dtype is applied, so x in f4 beside y and z in f8 under no stated dtype aborts rather than being promoted to cover both.
│   ├── calls PlyElement.describe(rows whose x is f4 while y and z are f8, 'vertex')
│   ├── calls PlyData.write(filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_columns_disagreeing_in_dtype_merge_under_a_stated_dtype_holding_them_all
│   ├── # A stated float64 casts the f4 column losslessly onto the width the f8 columns already hold, so the three merge into one xyz.
│   ├── calls PlyElement.describe(rows whose x is f4 while y and z are f8, 'vertex')
│   ├── calls PlyData.write(filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'dtype': 'float64'}}, device='cpu')
│   └── impls assert xyz is a [N, 3] float64 tensor holding the written coordinates
├── def test_no_filename_marker_decides_a_dtype
│   ├── # An uppercase _SEG basename never meant anything, and now neither does the lowercase one.
│   ├── calls torch.save(a [N, 4] float32 tensor, a filepath containing _SEG)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}}, device='cpu')
│   └── impls assert feat is still float32
├── def test_integer_xyz_is_rejected
│   ├── # An integer coordinate block is rejected rather than cast into a valid-looking float one.
│   ├── calls torch.save(a [N, 3] int64 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')  # the layout is stated so the integer coordinates are what fails
├── def test_windows_style_path_resolves
│   ├── # A path written with backslashes names the same file as the one written with slashes.
│   ├── calls write_ply(filepath)  # a ply's default layout assembles xyz with nothing stated, so the path is what the load tests
│   ├── impls windows_style_path = the filepath with its slashes turned into backslashes
│   ├── calls load_point_cloud(filepath=windows_style_path, device='cpu')
│   └── impls assert the result is a PointCloud
├── def test_sizes_from_one_point_upward
│   ├── # Point clouds of any row count load with their row count preserved.
│   └── for each size in 1, 10, 1000
│       ├── calls write_ply(filepath, num_points=size)
│       ├── calls load_point_cloud(filepath=filepath, device='cpu')
│       └── impls assert xyz is [size, 3]
├── def test_empty_point_cloud_is_rejected
│   ├── # A file holding no points is rejected rather than yielding an empty PointCloud.
│   ├── calls torch.save(a [0, 3] tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')  # the layout is stated so the empty file is what fails
└── def write_ply(filepath, num_points=8, extra_field=None, with_rgb=False, colour_dtype='uint8', colour_values=None)
    ├── # Writes a single-element PLY, since this suite needs a source that defines its own layout beside the .pth ones that define none.
    ├── impls columns = the x, y and z properties as float32  # impls-node-one-step:skip — names the three properties
    ├── if extra_field is not None
    │   └── impls columns gains extra_field as float32 holding the row indices  # whole numbers every narrower float width holds exactly, so a stated bfloat16 narrows them losslessly
    ├── if with_rgb
    │   └── impls columns gains red, green and blue in the ply dtype colour_dtype names, holding colour_values when given and mid-range values otherwise  # impls-node-one-step:skip — names the three properties
    ├── calls PlyElement.describe(the rows, 'vertex')
    └── calls PlyData.write(filepath)
```

`tests/utils/io/point_clouds/load_point_cloud/test_precision_handling.py`

```text
test_precision_handling.py
├── import numpy as np
├── import pytest
├── import torch
├── from plyfile import PlyData, PlyElement
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── def temp_dir
│   ├── # Yields a fresh temporary directory so each test writes its own files.
│   ├── # Every test below takes this fixture and joins a name onto it, so a bare filepath in one of them is that join rather than a name from nowhere.
│   └── impls yields the path of a tempfile.TemporaryDirectory
├── def test_an_f8_ply_gives_float64_coordinates
│   ├── # A reader neither widens nor narrows, so an f8 file's UTM-scale coordinates come back to within float64's own resolution without anything being asked for.
│   ├── calls write_float64_ply(filepath, coordinates of UTM magnitude)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert xyz is float64
│   └── impls assert the loaded coordinates match the written ones to 1e-9
├── def test_an_f4_ply_gives_float32_coordinates_and_their_loss
│   ├── # The same coordinates stored as f4 come back as float32 and carry that width's own loss, because the reader keeps the file's dtype rather than widening to cover it.
│   ├── calls write_float32_ply(filepath, coordinates of UTM magnitude)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert xyz is float32
│   └── impls assert the largest coordinate error exceeds 1e-3
├── def test_millimetre_offsets_survive_float64
│   ├── # Two points a millimetre apart at UTM magnitude stay distinguishable under float64.
│   ├── calls write_float64_ply(filepath, two coordinates a millimetre apart)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert the gap between the two loaded points is a millimetre to within 1e-9
├── def test_text_coordinates_keep_their_precision
│   ├── # The text reader parses in float64, so a long decimal survives the same way.
│   ├── impls a text file whose coordinates carry nine decimal places
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert the loaded coordinates match the written ones to 1e-9
├── def test_device_transfer_keeps_precision
│   ├── # Moving to another device changes where the values live and not what they are.
│   ├── impls skipped unless torch.cuda.is_available()
│   ├── calls write_float64_ply(filepath, coordinates of UTM magnitude)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── calls load_point_cloud(filepath=filepath, device='cuda')
│   └── impls assert the two agree once brought to the same device
├── def write_float64_ply(filepath, coordinates)
│   ├── # Writes a PLY whose x, y and z are stored as f8, so the width the reader hands back is the file's own.
│   ├── calls PlyElement.describe(the rows, 'vertex')
│   └── calls PlyData.write(filepath)
└── def write_float32_ply(filepath, coordinates)
    ├── # Writes the same coordinates as f4, so the same reader hands back the narrower width.
    ├── calls PlyElement.describe(the rows, 'vertex')
    └── calls PlyData.write(filepath)
```

`tests/utils/io/point_clouds/save_point_cloud/test_ply_saving.py`

```text
test_ply_saving.py
├── import pytest
├── import tempfile
├── import laspy
├── import numpy as np
├── import open3d as o3d
├── import torch
├── from plyfile import PlyData
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
├── @pytest.fixture def pc()
│   ├── # The in-memory cloud every case below saves, its coordinates handed in as one xyz block, whose columns the ply writer names x, y and z by ply's default layout.
│   ├── impls xyz = eight rows of coordinates as an [8, 3] float32 np.ndarray
│   ├── calls PointCloud(xyz=xyz, device='cpu')
│   └── return  # the cloud it built, whose target stands xyz on the one block it was handed as
├── def test_basic_ply_saving
│   ├── # Coordinates saved with nothing supplied are written by the ply default as x, y and z, and come back as the ones that were saved.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded xyz matches the saved coordinates
├── def test_an_in_memory_xyz_block_saves_as_x_y_z_with_nothing_supplied
│   ├── # An in-memory coordinate block maps back to the one name xyz, and the ply writer names such a block's columns x, y and z by ply's default layout, which are the three a ply reader assembles it from.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex columns are x, y and z, each stored as f4 and holding the saved coordinates
├── def test_saving_leaves_the_caller_s_cloud_as_it_was_handed_in
│   ├── # Save splits a copy into its output columns, so the cloud the caller handed in still holds xyz afterwards.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── impls assert pc.field_names() == ('xyz',)
│   └── impls assert pc.xyz still holds the coordinates it held before the save
├── def test_numpy_array_input
│   ├── # A PointCloud built from an np.array saves on the same terms as one built from a tensor.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded xyz matches the saved coordinates
├── def test_large_coordinates_precision
│   ├── # UTM-magnitude coordinates survive the round trip to within float32's own resolution.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the largest coordinate error is within torch.finfo(torch.float32).eps of the magnitude
├── def test_a_float_rgb_is_written_on_the_integer_range_its_meta_data_names
│   ├── # A float rgb declares 0 to 1 and a uint8 target declares 0 to 255, so save converts between the two conventions the dtypes name.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc = a cloud whose float rgb holds 0.0, 128/255 and 1.0, which are the 0-to-1 counterparts of whole 0-to-255 steps
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the red, green and blue columns hold 0, 128 and 255
├── def test_an_integer_rgb_is_written_on_its_own_range_untouched
│   ├── # A uint8 rgb saved under a uint8 target is already on the target range, so no conversion happens at all.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.rgb = an [8, 3] uint8 tensor holding 0, 128 and 255
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the columns hold exactly the values that were saved
├── def test_a_las_loaded_colour_saved_with_nothing_supplied_lands_on_i4_columns_at_its_own_values
│   ├── # The writer reads each column's ply dtype off its tensor's own dtype, so a uint16 colour parked in an int32 tensor is written as i4 at the values it came in with, with no rescaling at all.
│   ├── impls las_path = the path of a tempfile.NamedTemporaryFile with suffix '.las'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_las(las_path, with_rgb=True)
│   ├── calls load_point_cloud(filepath=las_path, device='cpu')
│   ├── impls assert the stored rgb tensor is torch.int32
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the red, green and blue columns are stored as i4
│   └── impls assert they hold exactly the uint16 values the las carried
├── def test_a_las_colour_saved_with_nothing_supplied_no_longer_loads_as_a_colour
│   ├── # An int32 colour names no convention, so the i4 red, green and blue columns that save wrote assemble into an rgb the load refuses.
│   ├── impls las_path = the path of a tempfile.NamedTemporaryFile with suffix '.las'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_las(las_path, with_rgb=True)
│   ├── calls load_point_cloud(filepath=las_path, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_a_las_loaded_colour_saves_as_u1_under_a_stated_uint8_dtype
│   ├── # A uint16 colour sits in an int32 tensor and save converts it off the range the colour MEANS, so a caller wanting u1 colours states uint8 and gets them when the values sit on uint8's grid.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a las whose colours are uint16 multiples of 257, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath, meta_data={'rgb': {'dtype': 'uint8'}})
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the red, green and blue columns are stored as u1
│   └── impls assert they hold the values rescaled from 0 to 65535, not from int32's own range
├── def test_a_uint16_las_colour_round_trips_through_its_own_range
│   ├── # A uint16 colour spans 0 to 65535, which is what las stores, and converting it to a uint8 target uses that range rather than a guessed one.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc = a cloud whose uint16 rgb holds 0, 32896 and 65535, each a whole multiple of 257 and so exactly recoverable
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the columns hold 0, 128 and 255
├── def test_a_signed_colour_saves_through_its_own_range
│   ├── # A signed integer colour spans that dtype's own range, so an int8 colour maps onto a u1 column by the offset the range mapping gives rather than being read as unsigned.
│   ├── impls pc = a cloud whose rgb entered as int8 holding -128, 0 and 127
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the colour columns hold 0, 128 and 255
├── def test_a_colour_off_the_target_grid_is_refused_at_save
│   ├── # A file is read back, so a colour that rounds into the target grid would return as a different colour than the one saved, and save refuses rather than writing it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc = a cloud whose uint16 rgb holds 1, which converts to 0 and back to 0
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
├── def test_a_colour_on_the_target_grid_is_written
│   ├── # The same pair of conventions carries a value that does convert back exactly, so what save refuses is the loss rather than the narrowing.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc = a cloud whose uint16 rgb holds 257, which converts to 1 and back to 257
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the columns hold 1
├── def test_float32_colours_on_the_uint8_grid_save_as_u1_under_a_stated_uint8
│   ├── # A colour's conversion is checked back at the source's own storage, so float32 values k/255 reach uint8 exactly and a caller stating uint8 gets u1 columns under the default layout.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.rgb = an [8, 3] float32 tensor of values k/255 for whole k from 0 to 255
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8'}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the red, green and blue columns are stored as u1 and hold each k
├── def test_a_float_colour_off_the_uint8_grid_is_refused_at_save
│   ├── # 0.5 lands between two uint8 steps, so converting it back misses the float32 value it came from and the conversion refuses it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.rgb = an [8, 3] float32 tensor holding 0.5
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8'}})
├── def test_an_ordinary_field_narrowing_out_of_range_is_refused_too
│   ├── # A colour reaches its target through a range mapping and an ordinary field through a dtype cast, and both refuse the value they cannot carry back.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a cloud whose uint16 intensity holds 300, filepath, meta_data={'intensity': {'dtype': 'uint8', 'layout': ('intensity',)}})
├── def test_a_colour_target_naming_no_convention_is_refused
│   ├── # Conventions are named by dtype and int64 names none, so a caller stating an int64 rgb target is refused when save brings the copy onto it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.rgb = an [8, 3] uint8 tensor holding 0, 128 and 255
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'int64', 'layout': ('red', 'green', 'blue')}})
├── def test_a_float_rgb_outside_zero_to_one_is_refused
│   ├── # What save asserts is that the values sit inside the range the field's current dtype declares.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose float rgb holds a value above one, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
├── def test_a_caller_layout_names_the_output_column_a_field_is_written_as
│   ├── # A caller's layout renames the output column, so a one-column feat lands under the name the caller gives it in place of its own.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.feat = an [8, 1] float32 tensor
│   ├── calls save_point_cloud(pc, filepath, meta_data={'feat': {'dtype': 'float32', 'layout': ('intensity',)}})
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the vertex data carries an intensity column with the saved values
│   └── impls assert the vertex data carries no feat column
├── def test_a_multi_column_field_takes_one_meta_data_name_per_column
│   ├── # A field of several columns is written under the several names its layout gives, one column per name.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.feat = an [8, 3] float32 tensor
│   ├── calls save_point_cloud(pc, filepath, meta_data={'feat': {'dtype': 'float32', 'layout': ('a', 'b', 'c')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex data carries the columns a, b and c with the saved values
├── def test_a_layout_naming_the_wrong_number_of_columns_is_refused
│   ├── # A mapping that does not name exactly as many source columns as the field carries leaves save with no names to write under.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose feat carries three columns, filepath, meta_data={'feat': {'dtype': 'float32', 'layout': ('a', 'b')}})
├── def test_a_multi_column_field_with_no_named_columns_is_refused_at_save
│   ├── # A ply column holds one value per point and an in-memory feat stands for its whole block under one name, so saving it with no columns named for it leaves three values against one column.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.feat = an [8, 3] float32 tensor
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath)
├── def test_a_field_the_meta_data_does_not_name_takes_its_target_from_itself
│   ├── # The meta data is what construction saw, so a field assigned afterwards is outside it and writes under its own name in the width it means, with the meta data the caller states the only way to name other columns.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc = a cloud loaded from a ply, then assigned a one-column uint8 mask under the name 'visible'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the file carries a visible column stored as u1
├── def test_a_field_the_meta_data_names_that_the_cloud_dropped_is_not_written
│   ├── # Deleting a field leaves the meta data naming it, and save walks the cloud's own fields, so the departed field reaches no column.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc = a cloud loaded from a ply carrying an intensity column, with its intensity field then deleted
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the file carries no intensity column
├── def test_a_pcd_loaded_field_needs_a_layout_to_reach_ply_columns
│   ├── # A pcd attribute is one named block, so xyz assembled from positions maps back to that one name, which ply's default layout names no columns for, and the caller names the columns to write under.
│   ├── impls pcd_path = the path of a tempfile.NamedTemporaryFile with suffix '.pcd'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_pcd(pcd_path, with_colors=True)
│   ├── calls load_point_cloud(filepath=pcd_path, meta_data={'xyz': {'layout': ('positions',)}, 'rgb': {'layout': ('colors',)}}, device='cpu')
│   ├── with pytest.raises(AssertionError)
│   │   └── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls save_point_cloud(the loaded cloud, filepath, meta_data={'xyz': {'layout': ('x', 'y', 'z')}, 'rgb': {'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the file carries x, y, z, red, green and blue columns
├── def test_a_las_sourced_ply_leads_with_x_y_z
│   ├── # las hands its coordinates over after its other dimensions, and coordinates lead the fields whatever order the source's columns arrive in, so the saved file's first columns are x, y and z.
│   ├── impls las_path = the path of a tempfile.NamedTemporaryFile with suffix '.las'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_las(las_path, with_rgb=True)
│   ├── calls load_point_cloud(filepath=las_path, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex columns open with x, y and z in that order
├── def test_a_pcd_sourced_ply_leads_with_x_y_z
│   ├── # Open3D may hand colors before positions, and the coordinates still lead the fields, so the saved file's first columns are x, y and z.
│   ├── impls pcd_path = the path of a tempfile.NamedTemporaryFile with suffix '.pcd'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_pcd(pcd_path, with_colors=True)
│   ├── calls load_point_cloud(filepath=pcd_path, meta_data={'rgb': {'layout': ('colors',)}, 'xyz': {'layout': ('positions',)}}, device='cpu')  # rgb is stated first, so the order of the caller's entries is covered too
│   ├── calls save_point_cloud(the loaded cloud, filepath, meta_data={'rgb': {'layout': ('red', 'green', 'blue')}, 'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex columns open with x, y and z in that order
├── def test_a_layout_repeating_a_column_is_refused
│   ├── # A caller-stated layout never passes through a meta data entry, so its own distinctness is checked at the door it comes in by.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'feat': {'layout': ('a', 'a', 'b')}})
├── def test_two_fields_writing_one_column_are_refused
│   ├── # Two layouts naming the same ply column would silently overwrite each other and then die inside numpy on a duplicate field name.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'feat': {'layout': ('shared',)}, 'label': {'layout': ('shared',)}})
├── def test_a_dtype_naming_no_such_field_is_refused_at_save
│   ├── # Save walks the cloud's own fields, so a key naming none of them would leave the meta data silently in force.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'nosuchfield': {'dtype': 'float32'}})
├── def test_a_layout_naming_no_such_field_is_refused
│   ├── # A name the cloud does not carry aborts rather than leaving the meta data's own columns silently in force.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'nosuchfield': {'layout': ('a',)}})
├── def test_a_value_that_is_not_a_point_cloud_is_refused
│   ├── # What this door refuses is a value that is not a PointCloud at all, before any default layout is written over its meta data.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a str, filepath)
├── def test_wrong_file_extension_error
│   ├── # An output path no writer owns is rejected the same way the load door rejects one, so a caller meets one behaviour across the module.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, a filepath that is not .ply)
├── def test_cuda_tensor_saving
│   ├── # Fields living on a cuda device are written from there, where one is available.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls skipped unless torch.cuda.is_available()
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded xyz matches the saved coordinates
├── def test_mixed_tensor_types_saving
│   ├── # One PointCloud may carry numpy and torch fields together.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded fields carry both
├── def test_a_pth_loaded_field_writes_back_under_the_index_names_its_meta_data_holds
│   ├── # The reverse mapping writes the source column names, and a .pth names its columns by position, so a ply written from one carries columns called 0, 1 and 2 until a caller states otherwise.
│   ├── impls pth_path = the path of a tempfile.NamedTemporaryFile with suffix '.pth'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls torch.save(a [N, 3] float32 tensor, pth_path)
│   ├── calls load_point_cloud(filepath=pth_path, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the file's columns are named 0, 1 and 2
│   ├── calls save_point_cloud(the loaded cloud, filepath, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the file's columns are named x, y and z  # naming them for another reader is the caller's to ask for, since the source never called them that
├── def test_save_load_round_trip
│   ├── # Across coordinate magnitudes, and with a feature column or with coordinates alone, saving then loading preserves the values.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls parametrized over small coordinates, UTM-magnitude coordinates, and coordinates with a feature column
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded xyz matches the saved coordinates
├── def test_indices_survive_the_ply_round_trip
│   ├── # A PointCloud carrying int64 indices comes back carrying their values, since the writer sends an int64 tensor to i4 with the values deciding.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.indices = the int64 tensor 0 through 7
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   ├── impls assert the loaded indices hold the saved values
│   └── impls assert the loaded indices are torch.int32  # ply carries no 64-bit integer, so the width that comes back is the one its i4 column names
├── def test_an_int64_target_goes_to_an_i4_column
│   ├── # ply has no 64-bit integer, so the ply column takes the same narrowing rule torch storage uses: the largest narrower dtype it carries, with the values deciding.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.label = an [8, 1] int32 tensor of small values
│   ├── calls save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'int64'}})
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the label column is stored as i4
│   └── impls assert it holds the values it was given
├── def test_a_uint64_target_is_refused_at_save
│   ├── # uint64 is no dtype torch storage carries, so a target stating one is refused when save brings the copy onto it, the way the constructor and load doors refuse it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls pc.label = an [8, 1] int32 tensor of small values
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'uint64'}})
├── def test_an_int64_target_whose_values_exceed_i4_is_refused
│   ├── # ply carries no int64, so this aborts inside the NARROWING: the target is narrowed to i4 and the values are tested against that candidate.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose label exceeds int32, filepath, meta_data={'label': {'dtype': 'int64'}})
├── def test_a_cast_that_would_lose_a_value_is_refused
│   ├── # ply carries i4 outright, so nothing narrows and this aborts inside the CAST instead — the same refusal reached by the other of the two paths a target can take.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose label exceeds int32, filepath, meta_data={'label': {'dtype': 'int32', 'layout': ('label',)}})
├── def test_a_u2_loaded_column_is_written_back_as_i4_at_its_own_values
│   ├── # The writer reads each column's ply dtype off its tensor's own dtype, so a field loaded from a u2 column and stored as int32 is written as i4 holding the values it started with.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a ply carrying a u2 label column, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the label column is stored as i4 and holds the values it started with
├── def test_a_u4_loaded_column_is_written_back_as_i4_at_its_own_values
│   ├── # The same one width up: a u4 column stored as int64 is written as the i4 an int64 tensor goes to, with the values deciding.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a ply carrying a u4 id column whose values sit inside int32's range, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the id column is stored as i4 and holds the values it started with
├── def test_a_u4_value_beyond_int32_is_refused_at_save
│   ├── # A u4 column's int64 tensor goes to i4, so a value above int32's range is one that column cannot hold and the cast refuses it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a ply carrying a u4 id column holding 4000000000, device='cpu')
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(the loaded cloud, filepath)
├── def test_xyz_is_written_in_the_width_its_meta_data_names
│   ├── # Coordinates are no longer written as f4 whatever they are: an f8 source goes back out as f8.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a ply whose x, y and z are f8, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the x, y and z columns are stored as f8
├── def test_precision_consistency_save_load
│   ├── # The precision the writer keeps is the precision the reader hands back.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the largest coordinate error is within torch.finfo(torch.float32).eps of the magnitude
├── def test_a_bool_field_widens_to_the_unsigned_byte_column
│   ├── # ply declares no boolean type at all, so a mask reaches a one-byte integer column, and u1 is the one of the two whose signedness matches bool's.
│   ├── impls pc = a cloud carrying a bool visible field
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the visible column is stored as u1 and holds the ones and zeros the mask carried
├── def test_a_bool_field_comes_back_from_ply_as_an_integer
│   ├── # The widening above is the one place a ply round trip does not return the dtype it took, and the record stays truthful about what the file it just read actually holds.
│   ├── impls pc = a cloud carrying a bool visible field
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls loaded = the cloud it loaded
│   ├── impls assert the loaded visible field is torch.uint8 and loaded.meta_data['visible'] == {'dtype': 'uint8', 'field': 'visible'}
│   └── impls assert its values equal the mask read as ones and zeros
├── def test_a_bfloat16_field_widens_to_leave_torch
│   ├── # numpy holds no bfloat16 at all, so the field widens to the narrowest numpy dtype containing it before it can become a ply column.
│   ├── impls pc = a cloud carrying a bfloat16 feature field
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the feature column is stored as f4 and holds the values bfloat16 carried
├── def write_las(filepath, num_points=8, with_rgb=False)
│   ├── # Writes a LAS carrying xyz plus, on request, the uint16 colour dimensions this suite saves back out, since a helper in another module is not in scope here.
│   ├── calls laspy.LasData(a header of the point format that carries the requested dimensions)
│   ├── impls las_data = the file it built
│   ├── impls the x, y and z dimensions of las_data  # impls-node-one-step:skip — names the three dimensions
│   ├── if with_rgb
│   │   └── impls the red, green and blue dimensions as mid-range uint16 values
│   └── calls las_data.write(filepath)
└── def write_pcd(filepath, num_points=8, with_colors=False)
    ├── # Writes a PCD through Open3D's tensor IO, so this suite can save a cloud whose fields each carry one attribute name.
    ├── calls o3d.t.geometry.PointCloud(a positions tensor of num_points rows)
    ├── if with_colors
    │   └── impls the colors attribute of that point cloud
    └── calls o3d.t.io.write_point_cloud(filepath, the point cloud)
```
