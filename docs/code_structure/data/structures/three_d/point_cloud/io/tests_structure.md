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
│   └── impls assert the meta data of red holds the layout ('red',)
├── def test_a_layout_assembles_the_columns_it_names
│   ├── # The layout half chooses which source columns a field is assembled from, which is the control that replaced renaming a column to feat.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'feat': {'layout': ('intensity',)}}, device='cpu')
│   ├── impls assert feat is present
│   ├── impls assert the meta data of feat holds the layout ('intensity',)
│   └── impls assert intensity is absent  # a caller-stated layout CONSUMES its columns, so the column does not also survive under its own name
├── def test_a_field_s_meta_data_names_the_source_columns_it_was_assembled_from
│   ├── # A PLY maps its three coordinate columns onto xyz and its three color columns onto rgb, and the meta data keeps that mapping.
│   ├── calls write_ply(filepath, with_rgb=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the meta data of xyz holds the layout ('x', 'y', 'z')
│   └── impls assert the meta data of rgb holds the layout ('red', 'green', 'blue')
├── def test_a_source_defining_no_layout_is_assembled_by_the_caller
│   ├── # A .pth names nothing about its block, so its columns come back under their own indices and the caller's layout says which of them each field is assembled from.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'label': {'layout': ('3',)}}, device='cpu')
│   ├── impls assert the meta data of xyz holds the layout ('0', '1', '2')
│   ├── impls assert the meta data of label holds the layout ('3',)
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
│   ├── # Every format's layout starts from its coordinate columns, so a file that names none of them is not a point cloud this reader can build.
│   ├── calls write_ply(filepath, without_coordinates=True)
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
├── def test_a_pcd_without_positions_is_refused
│   ├── # Open3D names the coordinate attribute positions, so a pcd carrying none of it is not a point cloud this reader can build.
│   ├── calls write_pcd(filepath, without_positions=True)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_a_las_bit_packed_dimension_enters_as_an_ordinary_uint8_field
│   ├── # laspy materializes a bit-packed dimension as uint8, so uint8 is what enters the obj and what the meta data stores, with no special treatment.
│   ├── calls write_las(filepath, with_classification=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the return_number field is torch.uint8
│   └── impls assert its meta data entry holds 'uint8' and the layout ('return_number',)
├── def test_a_las_uint16_colour_keeps_its_own_width
│   ├── # las stores colours as uint16, and a reader neither widens nor narrows, so the meta data holds uint16 while torch parks it in int32.
│   ├── calls write_las(filepath, with_rgb=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the rgb field is torch.int32
│   ├── impls assert the meta data of rgb holds 'uint16'
│   ├── impls assert the meta data entry for rgb holds 'uint16'  # the meta data and the storage differ here, which is the divergence a display reads the wrong side of when it reads the tensor
│   └── impls assert the meta data of rgb holds the layout ('red', 'green', 'blue')
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
│   ├── impls assert xyz holds the real-world coordinates the file was written with, not the integers the dimensions store
│   └── impls assert the meta data of xyz holds 'float64' and the layout ('x', 'y', 'z')  # the coordinate's own dtype is the scaled float and its name is the one laspy gives it, while the raw int32 dimension is the container's storage of both, the same divergence a uint16 colour has inside an int32 tensor
├── def test_a_laz_file_loads_as_its_las_counterpart_does
│   ├── # The compressed container changes nothing about the dimensions laspy hands back.
│   ├── calls write_las(filepath, with_rgb=True, compressed=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the coordinates match the written ones
│   └── impls assert the meta data of rgb holds 'uint16'
├── def test_a_pcd_names_its_own_layout_from_its_attributes
│   ├── # A pcd's attributes are named, so each one is a source column in its own right and the reader states the layout over those names.
│   ├── calls write_pcd(filepath, with_colors=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert xyz is [N, 3] and rgb is [N, 3]
│   └── impls assert the meta data of xyz holds the layout ('positions',) and the meta data of rgb holds ('colors',)
├── def test_a_pcd_attribute_beyond_positions_and_colors_keeps_its_own_name
│   ├── # Only the two Open3D names for coordinates and colour are renamed; every other attribute names its own field.
│   ├── calls write_pcd(filepath, with_colors=True, extra_attribute='intensity')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert the loaded fields are xyz, rgb and intensity
├── def test_a_pcd_colour_arrives_as_uint8_whatever_the_writer_held
│   ├── # PCD stores colour as bytes, so Open3D scales a float colour by 255 on the way out and the field that comes back means the 0-to-255 convention rather than the 0-to-1 one it was written in.
│   ├── calls write_pcd(filepath, with_colors=True, colors_dtype='float32')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the meta data of rgb holds the source dtype 'uint8'  # the meta data states the file, not the writer's intent, which is what keeps the convention read off it truthful
│   └── impls assert rgb spans 0 to 255 rather than 0 to 1
├── def test_a_pcd_unsigned_attribute_reaches_the_cloud_at_its_own_width
│   ├── # Open3D carries unsigned widths torch has none of, so the reader's route to numpy is what decides whether a uint16 attribute arrives as uint16 or not at all.
│   ├── calls write_pcd(filepath, with_colors=True, extra_attribute='intensity', extra_attribute_dtype='uint16')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the meta data of intensity holds the source dtype 'uint16'
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
│   ├── impls assert xyz is [N, 3]
│   └── impls assert the meta data of xyz holds the layout ('vertex.x', 'vertex.y', 'vertex.z')
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
│   ├── # An OFF file's vertex block loads as coordinates.
│   ├── calls write_off(filepath, four vertices)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz is [4, 3]
├── def test_off_keeps_only_the_leading_three_columns
│   ├── # A vertex line carrying more than three numbers contributes only its coordinates.
│   ├── calls write_off(filepath, vertices of six numbers each)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert xyz is [N, 3]
│   └── impls assert xyz holds the leading three of each line
├── def test_off_coordinates_beyond_float32_are_refused
│   ├── # float32 is the width this format is read at, so a vertex float32 cannot hold exactly aborts rather than the read widening to cover it.
│   ├── calls write_off(filepath, a vertex whose coordinate is not exactly representable in float32)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_off_without_its_header_is_rejected
│   ├── # A file whose first line is not OFF is rejected rather than parsed as vertices.
│   ├── calls write_off(filepath, four vertices, header='INVALID')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_an_off_with_its_counts_glued_to_the_keyword_loads
│   ├── # ModelNet40 writes the keyword and the counts on one line, and refusing that shape would refuse the dataset this design loads .off for.
│   ├── calls write_off(filepath, four vertices, glue_counts_to_header=True)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz is [4, 3] and holds the written coordinates
├── def test_an_off_with_a_comment_before_its_counts_loads
│   ├── # OFF permits a comment line, so the counts are the next line that carries any, rather than the next line whatever it holds.
│   ├── calls write_off(filepath, four vertices, comment='made by something')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz is [4, 3] and holds the written coordinates
├── def test_load_from_ply_returns_columns_and_their_layout
│   ├── # The PLY reader hands back the file's own columns and how PLY names their layout; assembly into xyz and rgb happens later.
│   ├── calls write_ply(filepath, with_rgb=True, extra_field='intensity')
│   ├── calls _load_from_ply(filepath=filepath)
│   ├── impls columns, layout = what it returned
│   ├── impls assert the keys of columns are x, y, z, red, green, blue and intensity  # impls-node-one-step:skip — names the keys
│   ├── impls assert each column is an np.ndarray in the dtype the file stored it in
│   ├── impls assert layout maps xyz to ('x', 'y', 'z') and rgb to ('red', 'green', 'blue')
│   └── impls assert layout maps intensity to ('intensity',)
├── def test_load_from_txt_returns_its_columns_under_their_indices
│   ├── # The text reader names its columns by position and nothing else, so a seven-column file hands back seven float64 arrays and no field names at all.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls _load_from_txt(filepath=filepath)
│   ├── impls assert its keys are '0' through '6'
│   ├── impls assert every column is float64
│   └── impls assert the layout it returned is None  # the file defines none, which is what makes the caller's meta data required rather than optional
├── def test_load_from_txt_divines_no_fields_from_the_column_count
│   ├── # Seven columns named xyz, rgb and feat was the reader's own invention, and the width now decides nothing, which is what the dataset's stated meta data replaced.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls _load_from_txt(filepath=filepath)
│   └── impls assert its keys carry no xyz, no rgb and no feat
├── def test_load_from_txt_reads_columns_however_they_are_spaced
│   ├── # Point cloud text is written aligned as often as it is written with single spaces, so the reader splits on runs of whitespace rather than on one named space character.
│   ├── calls write_txt(filepath, num_columns=3, spacing='aligned')
│   ├── calls _load_from_txt(filepath=filepath)
│   ├── impls assert it returned three columns of the written values
│   ├── calls write_txt(filepath, num_columns=3, spacing='single')
│   ├── calls _load_from_txt(filepath=filepath)
│   └── impls assert it returned the same three columns  # one delimiter reads both files, where naming a single space reads only the second
├── def test_a_txt_of_one_row_still_has_columns_to_key
│   ├── # numpy drops the column axis for a file holding one row, so a single-point cloud would arrive with scalars where the columns keyed by index should be.
│   ├── calls write_txt(filepath, num_points=1, num_columns=3)
│   ├── calls _load_from_txt(filepath=filepath)
│   ├── impls assert its keys are '0', '1' and '2'
│   └── impls assert each of those columns holds exactly one entry
├── def test_load_from_pth_returns_a_field_dict
│   ├── # The .pth reader hands back a dict in whatever form the file was saved in.
│   ├── calls write_pth(filepath, a [N, 4] torch tensor)
│   ├── calls _load_from_pth(filepath=filepath)
│   └── impls assert its xyz is a torch.Tensor
├── def test_load_from_off_returns_a_field_dict
│   ├── # The OFF reader hands back a dict of plain float32 columns; device placement is PointCloud's, which is why the readers take no device.
│   ├── calls write_off(filepath, four vertices)
│   ├── calls _load_from_off(filepath=filepath)
│   ├── impls assert its columns are float32 np.ndarrays of shape [4]
│   └── impls assert the layout it returned is {'xyz': ('x', 'y', 'z')}
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
    ├── calls o3d.t.geometry.PointCloud(a positions tensor of num_points rows)
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
│   ├── impls assert xyz is float64
│   ├── impls assert feat is still float32
│   └── impls assert the meta data of xyz still holds 'float32'
├── def test_a_dtype_reaches_a_field_the_coordinates_are_not
│   ├── # The retired dtype argument could only cast the coordinates; the dtype half reaches any field by name.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'dtype': 'float64', 'layout': ('3',)}}, device='cpu')
│   ├── impls assert feat is float64
│   └── impls assert xyz is still float32
├── def test_a_column_no_layout_names_is_not_loaded
│   ├── # A caller writing the layout by hand has chosen which columns become fields, so a column none of them names is simply absent rather than assembled into a field of its own.
│   ├── calls torch.save(a [N, 7] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')
│   └── impls assert the loaded fields are xyz alone
├── def test_a_bfloat16_target_is_reached_from_a_float32_source
│   ├── # The dtype a caller states may name the one dtype only torch carries, and the cast runs in the system that has it.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'intensity': {'dtype': 'bfloat16'}}, device='cpu')
│   └── impls assert intensity is stored as torch.bfloat16
├── def test_a_dtype_the_caller_states_leaves_the_field_meaning_what_its_source_held
│   ├── # A stated dtype casts the values without rescaling them, so they stay on the source's convention and the record goes on being what the field means.
│   ├── calls write_ply(filepath, with_rgb=True, colour_columns='uint16')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'rgb': {'dtype': 'uint8'}}, device='cpu')
│   ├── impls assert the stored rgb tensor is torch.uint8
│   └── impls assert the meta data entry for rgb still holds 'uint16', which is the convention those values are on
├── def test_a_dtype_alone_is_enough_where_the_source_defines_the_layout
│   ├── # A ply names its own columns, so a caller who only wants a different dtype states only that and the file's layout stands.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, meta_data={'intensity': {'dtype': 'float64'}}, device='cpu')
│   ├── impls assert intensity is float64
│   └── impls assert the meta data of xyz still holds the layout ('x', 'y', 'z')
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
│   ├── # torch carries no uint16, so the field is stored in the narrowest torch dtype that holds it while the meta data keeps what the column held.
│   ├── calls PlyElement.describe(rows carrying a u2 label column, 'vertex')
│   ├── calls PlyData.write(filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the label field is torch.int32
│   └── impls assert the meta data of label holds 'uint16'
├── def test_a_u4_column_is_stored_as_int64_and_notes_uint32
│   ├── # The same patch one width up: torch carries no uint32 either, and the meta data still keeps the column's own dtype.
│   ├── calls PlyElement.describe(rows carrying a u4 id column, 'vertex')
│   ├── calls PlyData.write(filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the id field is torch.int64
│   └── impls assert the meta data of id holds 'uint32'
├── def test_a_field_whose_source_columns_disagree_in_dtype_is_rejected
│   ├── # A field is assembled only from columns that all hold one dtype; a file whose columns disagree aborts rather than being promoted to a dtype covering them all.
│   ├── calls PlyElement.describe(rows whose x is f4 while y and z are f8, 'vertex')
│   ├── calls PlyData.write(filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
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
│   ├── impls windows_style_path = the filepath with its slashes turned into backslashes
│   ├── calls load_point_cloud(filepath=windows_style_path, device='cpu')
│   └── impls assert the result is a PointCloud
├── def test_sizes_from_one_point_upward
│   ├── # Point clouds of any row count load with their row count preserved.
│   └── for each size in 1, 10, 1000
│       ├── calls load_point_cloud(filepath=filepath, device='cpu')
│       └── impls assert xyz is [size, 3]
├── def test_empty_point_cloud_is_rejected
│   ├── # A file holding no points is rejected rather than yielding an empty PointCloud.
│   ├── calls torch.save(a [0, 3] tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu')  # the layout is stated so the empty file is what fails
└── def write_ply(filepath, num_points=8, extra_field=None)
    ├── # Writes a single-element PLY, since this suite needs a source that defines its own layout beside the .pth ones that define none.
    ├── impls columns = the x, y and z properties as float32  # impls-node-one-step:skip — names the three properties
    ├── if extra_field is not None
    │   └── impls columns gains extra_field as float32
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
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
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
├── from data.structures.three_d.point_cloud.io import load_point_cloud, save_point_cloud
├── def test_basic_ply_saving
│   ├── # Coordinates written to a PLY under the column names their meta data entry gives them come back as the ones that were saved.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded xyz matches the saved coordinates
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
│   ├── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the columns hold exactly the values that were saved
├── def test_a_las_loaded_colour_round_trips_to_ply_with_nothing_supplied
│   ├── # The whole defect in one test: a uint16 colour loaded from las sits in an int32 tensor, and saving with no meta at all must write it back as u2 under the file's own column names, at the values it came in with.
│   ├── impls las_path = the path of a tempfile.NamedTemporaryFile with suffix '.las'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_las(las_path, with_rgb=True)
│   ├── calls load_point_cloud(filepath=las_path, device='cpu')
│   ├── impls assert the stored rgb tensor is torch.int32
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the red, green and blue columns are stored as u2
│   └── impls assert they hold exactly the uint16 values the las carried, with no rescaling at all
├── def test_a_las_loaded_colour_saves_through_its_own_range_with_no_dtype_stated
│   ├── # The whole colour defect: a uint16 colour sits in an int32 tensor, and save must read the range off what the field MEANS rather than off the tensor it is parked in.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a las whose colours are uint16 multiples of 257, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath, meta_data={'rgb': {'dtype': 'uint8'}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the red, green and blue columns hold the values rescaled from 0 to 65535, not from int32's own range
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
├── def test_an_ordinary_field_narrowing_out_of_range_is_refused_too
│   ├── # A colour reaches its target through a range mapping and an ordinary field through a dtype cast, and both refuse the value they cannot carry back.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a cloud whose uint16 intensity holds 300, filepath, meta_data={'intensity': {'dtype': 'uint8', 'layout': ('intensity',)}})
├── def test_a_colour_target_ply_cannot_carry_is_refused
│   ├── # A colour target's range IS the convention the conversion fills, so narrowing it would rewrite the convention rather than narrow a value; save refuses instead of scaling onto a range the column cannot hold.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(pc, filepath, meta_data={'rgb': {'dtype': 'int64', 'layout': ('red', 'green', 'blue')}})
├── def test_a_float_rgb_outside_zero_to_one_is_refused
│   ├── # What save asserts is that the values sit inside the range the field's current dtype declares.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose float rgb holds a value above one, filepath, meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}})
├── def test_a_field_is_written_under_the_column_names_its_meta_data_holds
│   ├── # The output column names come from the meta data's layout, never from the field name, so a one-column field lands under the name its source gave it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath, meta_data={'feat': {'dtype': 'float32', 'layout': ('intensity',)}})
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the vertex data carries an intensity column with the saved values
│   └── impls assert the vertex data carries no feat column
├── def test_a_multi_column_field_takes_one_meta_data_name_per_column
│   ├── # A field of several columns is written under the several names its layout gives, one column per name.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath, meta_data={'feat': {'dtype': 'float32', 'layout': ('a', 'b', 'c')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex data carries the columns a, b and c with the saved values
├── def test_a_layout_naming_the_wrong_number_of_columns_is_refused
│   ├── # A mapping that does not name exactly as many source columns as the field carries leaves save with no names to write under.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose feat carries three columns, filepath, meta_data={'feat': {'dtype': 'float32', 'layout': ('a', 'b')}})
├── def test_a_multi_column_identity_layout_is_refused_by_the_column_count
│   ├── # An in-memory three-column field gets one name standing for the whole block, which is one name against three ply columns.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises(AssertionError)
│       └── calls save_point_cloud(a pc whose xyz was built in memory under no layout, filepath)
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
│   ├── # A pcd attribute is one named block, so a cloud loaded from one meta data entrys one name against three ply columns exactly as an in-memory field does, and the caller names the columns to write under.
│   ├── impls pcd_path = the path of a tempfile.NamedTemporaryFile with suffix '.pcd'
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls write_pcd(pcd_path, with_colors=True)
│   ├── calls load_point_cloud(filepath=pcd_path, device='cpu')
│   ├── with pytest.raises(AssertionError)
│   │   └── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls save_point_cloud(the loaded cloud, filepath, meta_data={'xyz': {'layout': ('x', 'y', 'z')}, 'rgb': {'layout': ('red', 'green', 'blue')}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the file carries x, y, z, red, green and blue columns
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
│   ├── # The name a cloud is missing coordinates no longer describes anything reachable, since a PointCloud cannot be built without them; what this door still refuses is a value that is not one at all.
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
│   ├── # A PointCloud carrying indices comes back carrying them, in the width the meta data names.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   ├── impls assert the loaded indices hold the saved values
│   └── impls assert the loaded indices are torch.int64
├── def test_an_int64_target_goes_to_an_i4_column
│   ├── # ply has no 64-bit integer, so the ply column takes the same narrowing rule torch storage uses: the largest narrower dtype it carries, with the values deciding.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'int64'}})
│   ├── calls PlyData.read(filepath)
│   ├── impls assert the label column is stored as i4
│   └── impls assert it holds the values it was given
├── def test_a_uint64_target_goes_to_a_u4_column
│   ├── # An unsigned target keeps an unsigned column, which is what the signedness tie-break among equal-width narrowing candidates is for.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'uint64'}})
│   ├── calls PlyData.read(filepath)
│   └── impls assert the label column is stored as u4
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
├── def test_a_u2_column_round_trips_back_to_u2
│   ├── # A field loaded from a u2 column is stored as int32 and written back as u2, because save follows the meta data rather than the stored dtype.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a ply carrying a u2 label column, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the label column is stored as u2 and holds the values it started with
├── def test_a_u4_column_round_trips_back_to_u4
│   ├── # The same one width up: a u4 column stored as int64 is written back as u4, which is the whole point of noting the source dtype.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls load_point_cloud(a ply carrying a u4 id column, device='cpu')
│   ├── calls save_point_cloud(the loaded cloud, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the id column is stored as u4 and holds the values it started with
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
│   ├── # The widening above is the one place a ply round trip does not return the dtype it took, and the meta data stays truthful about what the file it just read actually holds.
│   ├── impls pc = a cloud carrying a bool visible field
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert the loaded visible field is torch.uint8 and its meta data entry holds 'uint8'
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
