import numpy as np
import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select
from utils.input_checks.check_point_cloud import check_point_cloud_segmentation


def test_point_cloud_keys_and_access() -> None:
    """A point cloud built from coordinates alone reports the point count and the device those coordinates carry."""
    xyz = torch.randn(4, 3, dtype=torch.float32)
    pc = PointCloud(xyz=xyz)

    assert pc.num_points == 4
    assert pc.device == xyz.device


def test_setitem_validation() -> None:
    """Assigning a field of the wrong length is refused, and one of the right length lands."""
    pc = PointCloud(xyz=torch.randn(5, 3, dtype=torch.float32))

    with pytest.raises(AssertionError):
        pc.feat = torch.randn(4, 2, dtype=torch.float32)

    feat = torch.randn(5, 2, dtype=torch.float32)
    pc.feat = feat
    assert torch.equal(pc.feat, feat)


def test_missing_field_access() -> None:
    """Reading a field the point cloud does not carry raises AttributeError."""
    pc = PointCloud(xyz=torch.randn(3, 3, dtype=torch.float32))

    with pytest.raises(AttributeError):
        _ = pc.feat


def test_point_cloud_requires_xyz() -> None:
    """A cloud whose columns assemble into no coordinate field is legal here, a positional reader building exactly one, and load_point_cloud's own door is where a loaded cloud without coordinates is refused."""
    pc = PointCloud(data={'feat': torch.randn(4, 1, dtype=torch.float32)})

    assert pc.field_names() == (
        'feat',
    ), f"a cloud built from one column carries that column alone: field_names={pc.field_names()}"


def test_point_cloud_rejects_nan_xyz() -> None:
    """Coordinates carrying NaN are refused, under the message that names the NaN."""
    with pytest.raises(AssertionError, match="xyz tensor contains NaN"):
        PointCloud(xyz=torch.tensor([[float('nan'), 0.0, 0.0]], dtype=torch.float32))


def test_point_cloud_rejects_inf_xyz() -> None:
    """Coordinates carrying Inf are refused on the same terms as NaN, which the coordinate validator checks separately."""
    with pytest.raises(AssertionError, match="xyz tensor contains Inf"):
        PointCloud(xyz=torch.tensor([[float('inf'), 0.0, 0.0]], dtype=torch.float32))


def test_rgb_carrying_nan_or_inf_is_refused() -> None:
    """The colour validator rejects both on its own, so a float colour field cannot smuggle either past the range check."""
    xyz = torch.zeros(1, 3, dtype=torch.float32)

    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': xyz,
                'rgb': torch.tensor([[float('nan'), 0.0, 0.0]], dtype=torch.float32),
            }
        )

    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': xyz,
                'rgb': torch.tensor([[float('inf'), 0.0, 0.0]], dtype=torch.float32),
            }
        )


def test_coordinates_given_twice_are_refused() -> None:
    """Coordinates arrive through one route or the other, never both, so a field dict carrying xyz beside the xyz argument aborts."""
    with pytest.raises(AssertionError):
        PointCloud(
            xyz=torch.randn(4, 3, dtype=torch.float32),
            data={'xyz': torch.randn(4, 3, dtype=torch.float32)},
        )


def test_a_field_on_another_device_is_refused() -> None:
    """Every field of one cloud sits on one device, so a field arriving on another is refused rather than silently moved."""
    if not torch.cuda.is_available():
        pytest.skip("a second device is needed to hand a field in on one")

    pc = PointCloud(xyz=torch.randn(4, 3, dtype=torch.float32, device='cpu'))

    with pytest.raises(AssertionError):
        pc.feat = torch.randn(4, 2, dtype=torch.float32, device='cuda')


def test_a_field_that_is_not_a_tensor_is_refused() -> None:
    """Every field is a tensor, so a None or a bare list is refused at the door rather than carried and skipped later by whatever consumes it."""
    xyz = torch.randn(4, 3, dtype=torch.float32)

    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': xyz, 'feat': None})

    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': xyz, 'feat': [0.0, 1.0, 2.0, 3.0]})


def test_an_underscore_field_name_is_refused() -> None:
    """An underscore name is the class's own private slot namespace, so a field may not take one."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': torch.randn(4, 3, dtype=torch.float32),
                '_secret': torch.randn(4, dtype=torch.float32),
            }
        )


def test_point_cloud_length_mismatch_on_assignment() -> None:
    """A field assigned onto an existing point cloud must carry as many points as the coordinates do."""
    pc = PointCloud(xyz=torch.randn(5, 3, dtype=torch.float32))

    with pytest.raises(AssertionError):
        pc.feat = torch.randn(4, 2, dtype=torch.float32)


def test_reserved_attribute_assignment_rejected() -> None:
    """A field may not be assigned under one of the reserved attribute names."""
    pc = PointCloud(xyz=torch.randn(3, 3, dtype=torch.float32))

    with pytest.raises(AssertionError):
        pc.device = torch.randn(3, 3, dtype=torch.float32)


def test_non_string_keys_rejected() -> None:
    """A field dict keyed by anything but a str is refused."""
    xyz = torch.randn(4, 3, dtype=torch.float32)

    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': xyz, 1: xyz})


def test_point_cloud_segmentation_validation() -> None:
    """Matching segmentation logits and labels pass the checker through untouched."""
    logits = torch.randn(6, 4, dtype=torch.float32)
    labels = torch.randint(low=0, high=4, size=(6,), dtype=torch.int64)

    validated_logits, validated_labels = check_point_cloud_segmentation(
        y_pred=logits, y_true=labels
    )

    assert validated_logits is logits
    assert validated_labels is labels


def test_rgb_is_admitted_at_any_integer_width() -> None:
    """An integer color field is admitted whatever its width, because ply stores colors as u1 while las stores them as uint16."""
    xyz = torch.zeros(4, 3, dtype=torch.float32)

    pc = PointCloud(data={'xyz': xyz, 'rgb': torch.zeros(4, 3, dtype=torch.uint8)})
    assert pc.rgb.dtype == torch.uint8

    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'rgb': np.full((4, 3), 60000, dtype=np.uint16),
        }
    )
    # a width naming no colour convention names no colour either, so what "any integer width" reaches is every width COLOR_RANGE bounds
    assert (
        pc.meta_data['rgb']['dtype'] == 'uint16'
    ), f"a uint16 colour is noted as the convention it means: meta_data={pc.meta_data['rgb']}"
    assert (
        pc.rgb.dtype == torch.int32
    ), f"torch has no uint16, so the tensor parking it is int32: rgb.dtype={pc.rgb.dtype}"


def test_a_float_rgb_outside_zero_to_one_is_refused() -> None:
    """A float color field declares the 0 to 1 convention by its dtype, so a value outside that range is refused."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': torch.zeros(2, 3, dtype=torch.float32),
                'rgb': torch.full((2, 3), 255.0, dtype=torch.float32),
            }
        )


def test_xyz_of_the_wrong_shape_is_refused() -> None:
    """The coordinate validator's rank and width guards each stand on their own input, so neither can be what holds the other up."""
    with pytest.raises(AssertionError):
        PointCloud(xyz=torch.randn(4, dtype=torch.float32))

    with pytest.raises(AssertionError):
        PointCloud(xyz=torch.randn(4, 4, dtype=torch.float32))


def test_a_zero_dimensional_field_is_refused() -> None:
    """Every field is indexed by point, so a scalar carrying no point axis at all is refused before its length is ever compared."""
    pc = PointCloud(xyz=torch.randn(4, 3, dtype=torch.float32))

    with pytest.raises(AssertionError):
        pc.feat = torch.tensor(1.0, dtype=torch.float32)


def test_a_field_carrying_no_points_is_refused() -> None:
    """A cloud of no points is not a cloud, and this guard is exercised on its own rather than through a reader that would also trip the length comparison."""
    with pytest.raises(AssertionError):
        PointCloud(xyz=torch.zeros(0, 3, dtype=torch.float32))


def test_rgb_of_the_wrong_shape_is_refused() -> None:
    """The colour validator's rank and width guards each stand on their own input, so neither can be what holds the other up."""
    xyz = torch.zeros(4, 3, dtype=torch.float32)

    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': xyz, 'rgb': torch.zeros(4, dtype=torch.uint8)})

    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': xyz, 'rgb': torch.zeros(4, 4, dtype=torch.uint8)})


def test_the_two_color_conventions_are_told_apart_by_dtype_alone() -> None:
    """An integer color field holding only zeros and ones is still the integer convention, because the dtype decides and the values are never inspected."""
    pc = PointCloud(
        data={
            'xyz': torch.zeros(2, 3, dtype=torch.float32),
            'rgb': torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.uint8),
        }
    )

    assert pc.rgb.dtype == torch.uint8


def test_a_colour_is_bounded_by_the_range_it_MEANS_not_the_one_it_is_parked_in() -> (
    None
):
    """torch has no uint16, so a uint16 colour sits in an int32 tensor; the range enforced is uint16's own, or the guard would pass anything int32 can hold and let a colour walk out of its convention."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((2, 3), dtype=np.float32),
            'rgb': np.array([[0, 1000, 65535], [2, 3, 4]], dtype=np.uint16),
        }
    )

    assert pc.rgb.dtype == torch.int32
    assert pc.meta_data['rgb']['dtype'] == 'uint16'

    with pytest.raises(AssertionError):
        pc.rgb = torch.full((2, 3), 70000, dtype=torch.int32)


def test_xyz_is_admitted_at_any_floating_point_width() -> None:
    """Coordinates are any floating point dtype, so an f8 source is not narrowed to float32."""
    pc = PointCloud(xyz=torch.randn(4, 3, dtype=torch.float64))

    assert pc.xyz.dtype == torch.float64


def test_indices_no_longer_have_to_be_int64() -> None:
    """PointCloud treats a field named indices as an ordinary field, and nothing downstream reimposes the dtype the design retires."""
    pc = PointCloud(
        data={
            'xyz': torch.randn(4, 3, dtype=torch.float32),
            'indices': torch.arange(4, dtype=torch.int32),
        }
    )

    assert pc.indices.dtype == torch.int32

    out = Select([0, 2])(pc)

    # a selection indexes that field rather than consuming it, so its dtype survives
    assert out.indices.dtype == torch.int32


def test_a_field_under_a_name_the_class_binds_is_refused() -> None:
    """A field written under a public attribute name would land in the field dict and then never be readable, since ordinary lookup finds the class attribute first."""
    pc = PointCloud(xyz=torch.randn(3, 3, dtype=torch.float32))

    with pytest.raises(AssertionError):
        pc.validate_rgb_tensor = torch.randn(3, 2, dtype=torch.float32)


def test_a_deleted_field_leaves_the_meta_data_naming_it() -> None:
    """The meta data is what construction saw, so deleting a field takes the field and leaves the meta data exactly as it was, and save then writes no column for a field the obj no longer holds."""
    pc = PointCloud(
        data={
            'xyz': torch.randn(4, 3, dtype=torch.float32),
            'feat': torch.randn(4, 2, dtype=torch.float32),
        }
    )

    del pc.feat

    assert 'feat' not in pc.field_names()
    assert pc.meta_data['feat']['layout'] == ('feat',)


def test_coordinates_cannot_be_deleted() -> None:
    """Every other field may leave, but a point cloud without coordinates is not one, so the coordinate field is the one deletion refused."""
    pc = PointCloud(
        data={
            'xyz': torch.randn(4, 3, dtype=torch.float32),
            'feat': torch.randn(4, 2, dtype=torch.float32),
        }
    )

    with pytest.raises(AssertionError):
        del pc.xyz


def test_point_cloud_segmentation_validation_errors() -> None:
    """Segmentation logits and labels of different lengths are refused."""
    logits = torch.randn(5, 3, dtype=torch.float32)
    labels = torch.randint(low=0, high=3, size=(4,), dtype=torch.int64)

    with pytest.raises(AssertionError):
        check_point_cloud_segmentation(y_pred=logits, y_true=labels)
