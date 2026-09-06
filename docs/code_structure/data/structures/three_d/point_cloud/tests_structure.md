# Point Cloud Data Structure Tests Structure

## Tests implementation structure

`tests/data/structures/three_d/test_point_cloud.py`

```text
test_point_cloud.py
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
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
│   ├── # A field dict with no coordinates in it is refused.
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloud(data=a dict carrying feat alone)
├── def test_point_cloud_rejects_nan_xyz
│   ├── # Coordinates carrying NaN are refused, under the message that names the NaN.
│   └── with pytest.raises(AssertionError, match="xyz tensor contains NaN")
│       └── calls PointCloud(xyz=a [1, 3] tensor whose first entry is NaN)
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
└── def test_point_cloud_segmentation_validation_errors
    ├── # Segmentation logits and labels of different lengths are refused.
    ├── impls logits = a [5, 3] float32 random tensor
    ├── impls labels = a [4] int64 tensor of class ids
    └── with pytest.raises(AssertionError)
        └── calls check_point_cloud_segmentation(y_pred=logits, y_true=labels)
```

`tests/data/structures/three_d/point_cloud/test_select_random_select.py`

```text
test_select_random_select.py
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.random_select import RandomSelect
├── from data.structures.three_d.point_cloud.select import Select
├── def test_pointcloud_initialization
│   ├── # A point cloud built from a field dict reports its point count, its field names in coordinates-first order, and its coordinates.
│   ├── impls xyz = a [4, 3] float32 tensor
│   ├── impls feat = a [4, 2] float32 tensor
│   ├── calls PointCloud(data={'xyz': xyz, 'feat': feat})
│   ├── impls assert num_points is 4
│   ├── impls assert field_names() is ('xyz', 'feat')
│   └── impls assert xyz reads back
├── def test_select_with_legacy_mapping_input
│   ├── # Selecting by a plain index list carries every field down and records the taken indices.
│   ├── calls PointCloud(data={'xyz': a [5, 3] tensor, 'feat': a [5, 1] tensor})
│   ├── calls Select(indices=[0, 3])
│   ├── impls out = the selection applied to that point cloud
│   ├── impls assert out is a PointCloud
│   ├── impls assert its xyz is rows 0 and 3 of the original   # impls-node-one-step:skip
│   ├── impls assert its feat is rows 0 and 3 of the original  # impls-node-one-step:skip
│   └── impls assert its indices are the int64 tensor [0, 3]
├── @pytest.mark.parametrize def test_select_pointcloud(xyz_values, feat_values, indices, expected_xyz_indices, expected_feat_indices)  # over one coordinates / features / indices case
│   ├── # Each parametrized selection takes exactly the named rows of every field and records the indices it took.
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
│   ├── # An index list takes those rows of coordinates, colors and a classification field alike, and records them as the indices field.
│   ├── calls PointCloud(data={'xyz': a [5, 3] float64 tensor, 'rgb': a [5, 3] float64 tensor, 'classification': a [5] int64 tensor})
│   ├── calls Select([0, 2, 4])
│   ├── impls result = the selection applied to that point cloud
│   ├── impls assert its xyz is rows 0, 2 and 4 of the original             # impls-node-one-step:skip
│   ├── impls assert its rgb is rows 0, 2 and 4 of the original             # impls-node-one-step:skip
│   ├── impls assert its classification is rows 0, 2 and 4 of the original  # impls-node-one-step:skip
│   └── impls assert its indices are the int64 tensor [0, 2, 4]
├── def test_select_basic_tensor
│   ├── # An int64 index tensor on the point cloud's device selects exactly as an index list does.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor, 'rgb': a [3, 3] float64 tensor})
│   ├── impls indices_tensor = the int64 tensor [1, 2] on the point cloud's device
│   ├── calls Select(indices_tensor)
│   ├── impls result = the selection applied to that point cloud
│   ├── impls assert its xyz is rows 1 and 2 of the original  # impls-node-one-step:skip
│   └── impls assert its indices are indices_tensor
├── def test_select_empty_indices
│   ├── # Selecting no points at all is refused, because a point cloud carries at least one point.
│   ├── calls PointCloud(data={'xyz': a [2, 3] float64 tensor, 'rgb': a [2, 3] float64 tensor, 'classification': a [2] int64 tensor})
│   ├── calls Select([])
│   └── with pytest.raises(AssertionError)
│       └── impls applies the selection to that point cloud
├── def test_select_single_point
│   ├── # A one-entry selection hands back a one-point cloud whose color field keeps its trailing three columns.
│   ├── calls PointCloud(data={'xyz': a [3, 3] float64 tensor, 'rgb': a [3, 3] float64 tensor})
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
│   ├── calls PointCloud(data={'xyz': a [4, 3] float64 tensor, 'rgb': a [4, 3] float64 tensor})
│   ├── calls RandomSelect(percentage=0.5)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   ├── impls assert its num_points is half of four
│   ├── impls assert its rgb carries one row per selected point
│   ├── impls assert its indices carry one entry per selected point
│   └── impls assert its indices are int64
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
├── def test_random_select_percentage_range
│   ├── # A quarter of a twenty-point cloud is five points.
│   ├── calls PointCloud(xyz=a [20, 3] float64 tensor)
│   ├── calls RandomSelect(percentage=0.25)
│   ├── impls result = the selection applied to that point cloud under seed 42
│   └── impls assert its num_points is a quarter of twenty
└── def test_random_select_count_range
    ├── # A count of ten out of a twenty-point cloud is ten points.
    ├── calls PointCloud(xyz=a [20, 3] float64 tensor)
    ├── calls RandomSelect(count=10)
    ├── impls result = the selection applied to that point cloud under seed 42
    └── impls assert its num_points is 10
```
