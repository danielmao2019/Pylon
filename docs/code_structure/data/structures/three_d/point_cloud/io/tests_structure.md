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
├── def test_ply_name_feat_leaves_a_loaded_column_alone
│   ├── # name_feat fills feat only for a column not already loaded under its own name.
│   ├── calls write_ply(filepath, extra_field='intensity')
│   ├── calls load_point_cloud(filepath=filepath, name_feat='intensity', device='cpu')
│   ├── impls assert intensity is present
│   └── impls assert feat is absent
├── def test_ply_custom_element_name
│   ├── # A single element named anything but vertex is found without nameInPly being given.
│   ├── calls write_ply(filepath, element_name='points')
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz is [N, 3]
├── def test_txt_xyz_only
│   ├── # A three-column text file loads its coordinates and carries no feat.
│   ├── calls write_txt(filepath, num_columns=3)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert xyz is [N, 3]
│   └── impls assert feat is absent
├── def test_txt_slpccd_label_column
│   ├── # In the seven-column SLPCCD layout the label column alone becomes feat.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert feat is [N, 1]
│   └── impls assert feat holds the seventh column
├── def test_txt_middle_width_uses_every_trailing_column
│   ├── # Between four and six columns every column past the third becomes feat.
│   ├── calls write_txt(filepath, num_columns=5)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert feat is [N, 2]
├── def test_pth_tensor
│   ├── # A saved torch tensor splits into coordinates and features at the third column.
│   ├── calls write_pth(filepath, a [N, 4] torch tensor)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   ├── impls assert xyz is [N, 3]
│   └── impls assert feat is [N, 1]
├── def test_pth_ndarray
│   ├── # A saved numpy array is accepted on the same terms as a tensor.
│   ├── calls write_pth(filepath, a [N, 3] np.ndarray)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz is [N, 3]
├── def test_pth_rejects_a_non_array_payload
│   ├── # A .pth holding anything but a tensor or an array is rejected rather than half-read.
│   ├── calls write_pth(filepath, a dict)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
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
├── def test_off_without_its_header_is_rejected
│   ├── # A file whose first line is not OFF is rejected rather than parsed as vertices.
│   ├── calls write_off(filepath, four vertices, header='INVALID')
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
├── def test_load_from_ply_returns_a_field_dict
│   ├── # The PLY reader hands back a plain dict of fields, not a PointCloud, and takes no device.
│   ├── calls write_ply(filepath, with_rgb=True, extra_field='intensity')
│   ├── calls _load_from_ply(filepath=filepath)
│   ├── impls assert the result is a dict
│   ├── impls assert its keys are xyz, rgb and intensity  # impls-node-one-step:skip — names the keys
│   └── impls assert its xyz is an np.ndarray of dtype float64
├── def test_load_from_txt_returns_a_field_dict
│   ├── # The text reader hands back a plain dict of float64 arrays.
│   ├── calls write_txt(filepath, num_columns=7)
│   ├── calls _load_from_txt(filepath=filepath)
│   ├── impls assert its keys are xyz and feat  # impls-node-one-step:skip — names the keys
│   └── impls assert its xyz is float64
├── def test_load_from_pth_returns_a_field_dict
│   ├── # The .pth reader hands back a dict in whatever form the file was saved in.
│   ├── calls write_pth(filepath, a [N, 4] torch tensor)
│   ├── calls _load_from_pth(filepath=filepath, device='cpu')
│   └── impls assert its xyz is a torch.Tensor
├── def test_load_from_off_returns_a_field_dict
│   ├── # The OFF reader hands back a dict whose xyz is already a tensor on the requested device.
│   ├── calls write_off(filepath, four vertices)
│   ├── calls _load_from_off(filepath=filepath, device='cpu')
│   ├── impls assert its xyz is [4, 3]
│   ├── impls assert its xyz is float32
│   └── impls assert its xyz sits on the cpu
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
├── def write_ply(filepath, num_points=8, with_rgb=False, extra_field=None, element_name='vertex')
│   ├── # Writes a PLY whose one element carries xyz plus whichever optional columns the caller asks for.
│   ├── impls columns = the x, y and z fields as float32  # impls-node-one-step:skip — names the three fields
│   ├── if with_rgb
│   │   └── impls columns gains red, green and blue as uint8  # impls-node-one-step:skip — names the three fields
│   ├── if extra_field is not None
│   │   └── impls columns gains extra_field as float32
│   ├── calls PlyElement.describe(the rows, element_name)
│   └── calls PlyData.write(filepath)
├── def write_txt(filepath, num_points=8, num_columns=3)
│   ├── # Writes a whitespace-separated point cloud behind the two header lines the reader skips.
│   ├── impls two header lines
│   └── impls num_points rows of num_columns floats
├── def write_pth(filepath, array)
│   ├── # Saves one array as the single block a .pth point cloud holds.
│   └── calls torch.save(array, filepath)
└── def write_off(filepath, vertices, header='OFF')
    ├── # Writes an OFF file, the header being a parameter so a malformed one can be written too.
    ├── impls header line
    ├── impls the vertex, face and edge counts  # impls-node-one-step:skip — names the three counts
    └── impls one line per vertex
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
│   └── impls yields the path of a tempfile.TemporaryDirectory
├── def test_values_survive_the_load
│   ├── # The coordinates that come back are the ones that were saved.
│   ├── calls torch.save(a known [N, 3] tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert xyz equals the saved tensor
├── def test_placed_on_the_requested_device
│   ├── # Every field lands on the device the caller named.
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── for each name in the result's field names
│       └── impls assert that field's device type is cpu
├── def test_placed_on_cuda_when_asked
│   ├── # The same placement holds for a cuda device, where one is available.
│   ├── impls skipped unless torch.cuda.is_available()
│   ├── calls load_point_cloud(filepath=filepath, device='cuda')
│   └── impls assert xyz's device type is cuda
├── def test_dtype_applies_to_xyz_only
│   ├── # The dtype argument governs the coordinates and leaves every other field as stored.
│   ├── calls torch.save(a [N, 4] float32 tensor, filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)
│   ├── impls assert xyz is float64
│   └── impls assert feat is still float32
├── def test_segmentation_feature_column_becomes_int64
│   ├── # A file whose basename carries _seg has its feature column read as labels.
│   ├── calls torch.save(a [N, 4] tensor, a filepath containing _seg)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert feat is int64
├── def test_feature_column_is_left_alone_without_the_seg_marker
│   ├── # The same file without _seg in its basename keeps its feature column's dtype.
│   ├── calls torch.save(a [N, 4] tensor, a filepath without _seg)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert feat is float32
├── def test_uint16_field_becomes_int32
│   ├── # torch carries no uint16, so such a field arrives as the narrowest type that holds it.
│   ├── calls PlyElement.describe(rows carrying a uint16 label column, 'vertex')
│   ├── calls PlyData.write(filepath)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert the label field is torch.int32
├── def test_uppercase_seg_marker_leaves_the_feature_column_alone
│   ├── # The _seg marker is matched as stored, so an uppercase _SEG basename is not a segmentation file.
│   ├── calls torch.save(a [N, 4] float32 tensor, a filepath containing _SEG)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu')
│   └── impls assert feat is still float32
├── def test_integer_xyz_is_rejected
│   ├── # An integer coordinate block is rejected rather than cast into a valid-looking float one.
│   ├── calls torch.save(a [N, 3] int64 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls load_point_cloud(filepath=filepath, device='cpu')
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
└── def test_empty_point_cloud_is_rejected
    ├── # A file holding no points is rejected rather than yielding an empty PointCloud.
    ├── calls torch.save(a [0, 3] tensor, filepath)
    └── with pytest.raises(AssertionError)
        └── calls load_point_cloud(filepath=filepath, device='cpu')
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
│   └── impls yields the path of a tempfile.TemporaryDirectory
├── def test_float64_preserves_large_utm_coordinates
│   ├── # Under float64 a UTM-scale coordinate comes back to within float64's own resolution.
│   ├── calls write_float64_ply(filepath, coordinates of UTM magnitude)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)
│   └── impls assert the loaded coordinates match the written ones to 1e-9
├── def test_float32_loses_large_utm_coordinates
│   ├── # Under float32 the same coordinates lose resolution, which is why the dtype argument exists.
│   ├── calls write_float64_ply(filepath, coordinates of UTM magnitude)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float32)
│   └── impls assert the largest coordinate error exceeds 1e-3
├── def test_millimetre_offsets_survive_float64
│   ├── # Two points a millimetre apart at UTM magnitude stay distinguishable under float64.
│   ├── calls write_float64_ply(filepath, two coordinates a millimetre apart)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)
│   └── impls assert the gap between the two loaded points is a millimetre to within 1e-9
├── def test_text_coordinates_keep_their_precision
│   ├── # The text reader parses in float64, so a long decimal survives the same way.
│   ├── impls a text file whose coordinates carry nine decimal places
│   ├── calls load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)
│   └── impls assert the loaded coordinates match the written ones to 1e-9
├── def test_device_transfer_keeps_precision
│   ├── # Moving to another device changes where the values live and not what they are.
│   ├── impls skipped unless torch.cuda.is_available()
│   ├── calls write_float64_ply(filepath, coordinates of UTM magnitude)
│   ├── calls load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)
│   ├── calls load_point_cloud(filepath=filepath, device='cuda', dtype=torch.float64)
│   └── impls assert the two agree once brought to the same device
└── def write_float64_ply(filepath, coordinates)
    ├── # Writes a PLY whose x, y and z are stored as float64 so precision is the reader's to lose.
    ├── calls PlyElement.describe(the rows, 'vertex')
    └── calls PlyData.write(filepath)
```

`tests/utils/io/point_clouds/save_point_cloud/test_ply_saving.py`

```text
test_ply_saving.py
├── import pytest
├── import tempfile
├── import numpy as np
├── import torch
├── from plyfile import PlyData
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.io import load_point_cloud, save_point_cloud
├── def test_basic_ply_saving
│   ├── # Coordinates written to a PLY come back as the ones that were saved.
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
├── def test_rgb_colors_saving
│   ├── # An rgb field of [0, 1] floats is stored as [0, 255] uint8 and read back at that scale.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   ├── impls assert the loaded fields carry rgb
│   └── impls assert through np.testing.assert_allclose that the loaded rgb matches the saved colours scaled to uint8
├── def test_colors_field_mapping
│   ├── # A field named colors is written under the PLY's own red, green and blue names.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   ├── impls assert the loaded fields carry rgb
│   └── impls assert through np.testing.assert_allclose that the loaded rgb matches the saved colours scaled to uint8
├── def test_normalized_colors_conversion
│   ├── # Read from the file itself, a [0, 1] colour lands on its [0, 255] counterpart.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the red, green and blue columns hold the scaled values to within one step
├── def test_single_feature_saving
│   ├── # A one-column feature field is written under its own name.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex data carries that column with the saved values
├── def test_multiple_features_saving
│   ├── # A multi-column field is split into one PLY column per index, suffixed by it.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert the vertex data carries one suffixed column per index
├── def test_none_field_handling
│   ├── # A field whose value is None is skipped.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls PlyData.read(filepath)
│   └── impls assert that column is absent from the vertex data
├── def test_missing_xyz_field_error
│   ├── # A value of a type other than PointCloud is rejected.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises
│       └── calls save_point_cloud(a str, filepath)
├── def test_wrong_file_extension_error
│   ├── # An output path no writer owns is rejected.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   └── with pytest.raises
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
├── def test_save_load_round_trip
│   ├── # Across coordinate magnitudes, and with a feature column or with coordinates alone, saving then loading preserves the values.
│   ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
│   ├── impls parametrized over small coordinates, UTM-magnitude coordinates, and coordinates with a feature column
│   ├── calls save_point_cloud(pc, filepath)
│   ├── calls load_point_cloud(filepath)
│   └── impls assert the loaded xyz matches the saved coordinates
└── def test_precision_consistency_save_load
    ├── # The precision the writer keeps is the precision the reader hands back.
    ├── impls filepath = the path of a tempfile.NamedTemporaryFile with suffix '.ply'
    ├── calls save_point_cloud(pc, filepath)
    ├── calls load_point_cloud(filepath)
    └── impls assert the largest coordinate error is within torch.finfo(torch.float32).eps of the magnitude
```
