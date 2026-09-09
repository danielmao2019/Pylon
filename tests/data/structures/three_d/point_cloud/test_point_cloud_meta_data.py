import pickle

import numpy as np
import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud


def test_the_construction_meta_data_records_the_dtype_the_source_held() -> None:
    """A field entering as a numpy uint16 array notes uint16, whatever torch had to store it as."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'intensity': np.arange(4, dtype=np.uint16),
        }
    )

    assert pc.meta_data['intensity']['dtype'] == 'uint16'
    assert pc.intensity.dtype == torch.int32


def test_a_meta_data_handed_over_is_kept_exactly_as_it_arrived() -> None:
    """A record handed over is already resolved, so the constructor keeps it rather than deriving a second one from the tensors it comes with."""
    pc = PointCloud(
        data={
            'xyz': torch.zeros(4, 3, dtype=torch.float32),
            'feat': torch.zeros(4, dtype=torch.float32),
        },
        meta_data={
            'xyz': {'dtype': 'float32', 'layout': ('xyz',)},
            # a layout names one column per column the field carries, so a record handed over states as many names as the field it comes with
            'feat': {'dtype': 'float64', 'layout': ('a',)},
        },
    )

    assert pc.meta_data['feat']['dtype'] == 'float64'
    assert pc.meta_data['feat']['layout'] == ('a',)
    # the record says what the field means; the constructor casts nothing on its account
    assert pc.feat.dtype == torch.float32


def test_the_constructor_casts_nothing_a_caller_did_not_hand_it() -> None:
    """A caller wanting another dtype hands the field over in it, so the one cast here is the crossing into torch and a float64 field stays float64."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'feat': np.zeros(4, dtype=np.float64),
        }
    )

    assert pc.feat.dtype == torch.float64
    assert pc.meta_data['feat']['dtype'] == 'float64'


def test_a_half_stated_meta_data_entry_is_refused() -> None:
    """The meta data is one whole record, so an entry stating one half is refused here rather than half-derived alongside a record that was handed over."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={'xyz': np.zeros((4, 3), dtype=np.float32)},
            meta_data={'xyz': {'dtype': 'float32'}},
        )


def test_a_bool_field_is_carried_as_its_own_kind() -> None:
    """bool is a kind of its own in the dtype universe, not a narrow integer, so a mask field enters as bool and is noted as bool rather than being widened to one."""
    pc = PointCloud(
        data={
            'xyz': torch.zeros(4, 3, dtype=torch.float32),
            'visible': torch.tensor([True, False, True, False]),
        }
    )

    assert pc.visible.dtype == torch.bool
    assert pc.meta_data['visible']['dtype'] == 'bool'


def test_a_bool_colour_is_refused() -> None:
    """A colour spans a range its dtype declares, and two values are not a range to span, so bool is refused at the colour door while staying a legal field dtype elsewhere."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': torch.zeros(4, 3, dtype=torch.float32),
                'rgb': torch.zeros(4, 3, dtype=torch.bool),
            }
        )


def test_a_bfloat16_field_is_stored_without_a_numpy_detour() -> None:
    """bfloat16 is torch's alone, so a field entering as one is cast in torch rather than through a numpy dtype that does not exist."""
    pc = PointCloud(
        data={
            'xyz': torch.zeros(4, 3, dtype=torch.float32),
            'feat': torch.zeros(4, dtype=torch.bfloat16),
        }
    )

    assert pc.feat.dtype == torch.bfloat16
    assert pc.meta_data['feat']['dtype'] == 'bfloat16'


def test_a_meta_entry_stating_an_empty_layout_is_refused() -> None:
    """A field assembled from no columns at all is not a field, so the door refuses the entry rather than reading past it."""
    with pytest.raises(AssertionError):
        # both halves stated, so the empty layout is the only thing left to refuse
        PointCloud(
            data={'xyz': np.zeros((4, 3), dtype=np.float32)},
            meta_data={'xyz': {'dtype': 'float32', 'layout': ()}},
        )


def test_a_meta_entry_naming_a_half_the_design_has_no_slot_for_is_refused() -> None:
    """An entry carries a dtype half, a layout half or both, so a misspelled key aborts rather than being read past in silence."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={'xyz': np.zeros((4, 3), dtype=np.float32)},
            meta_data={'xyz': {'dytpe': 'float32'}},
        )


def test_a_meta_entry_stating_neither_half_is_refused() -> None:
    """An entry that names no dtype and no layout asks for nothing, so it is refused rather than silently ignored downstream."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={'xyz': np.zeros((4, 3), dtype=np.float32)},
            meta_data={'xyz': {}},
        )


def test_a_record_handed_over_names_the_source_columns_of_an_in_memory_field() -> None:
    """A record is where the columns an in-memory field is written back under are declared, the identity mapping being all the field itself can say."""
    pc = PointCloud(
        data={'xyz': np.zeros((4, 3), dtype=np.float32)},
        meta_data={'xyz': {'dtype': 'float32', 'layout': ('x', 'y', 'z')}},
    )

    assert pc.meta_data['xyz']['layout'] == ('x', 'y', 'z')


def test_a_record_says_what_a_field_means_without_moving_the_value() -> None:
    """The record and the storage are two questions, so a record naming float32 over a float64 tensor leaves the tensor exactly where it is."""
    pc = PointCloud(
        data={'xyz': np.zeros((4, 3), dtype=np.float64)},
        meta_data={'xyz': {'dtype': 'float32', 'layout': ('x', 'y', 'z')}},
    )

    assert pc.xyz.dtype == torch.float64
    assert pc.meta_data['xyz']['dtype'] == 'float32'


def test_an_in_memory_field_meta_the_identity_mapping() -> None:
    """A field handed in under a name and no layout gets that name standing for its whole block, rather than nothing."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'feat': np.zeros(4, dtype=np.float32),
        }
    )

    assert pc.meta_data['feat']['layout'] == ('feat',)
    assert pc.meta_data['xyz']['layout'] == ('xyz',)


def test_a_float128_field_is_stored_in_the_widest_float_torch_carries() -> None:
    """float128 is ruled in or out by whether its values survive the narrowing, not by its name, so a field whose values fit float64 enters and keeps the meta data entry for what its source held."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'feat': np.arange(4, dtype=np.float128),
        }
    )

    assert pc.feat.dtype == torch.float64
    assert pc.meta_data['feat']['dtype'] == 'float128'


def test_a_float128_field_whose_values_need_its_width_is_refused() -> None:
    """The same dtype is ruled out by the same test when the values do not survive, which is what makes the rule about values rather than names."""
    feat = np.full(4, np.float128(1) + np.float128(2) ** -63, dtype=np.float128)

    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': np.zeros((4, 3), dtype=np.float32), 'feat': feat})


def test_a_complex_field_is_stored_through_its_component_float() -> None:
    """A complex dtype is compared through the float its parts are, so a complex256 field whose parts fit float64 enters as complex128."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'feat': np.array([1 + 2j] * 4, dtype=np.complex256),
        }
    )

    assert pc.feat.dtype == torch.complex128
    assert pc.meta_data['feat']['dtype'] == 'complex256'


def test_complex_coordinates_are_refused() -> None:
    """Coordinates are a floating point field whatever the dtype universe admits elsewhere, so a complex block is refused at the coordinate door rather than the dtype one."""
    with pytest.raises(AssertionError):
        PointCloud(xyz=np.zeros((4, 3), dtype=np.complex128))


def test_a_uint64_source_is_refused() -> None:
    """uint64 is unsupported as a source dtype whatever values it carries, so no representability test is reached at all."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': np.zeros((4, 3), dtype=np.float32),
                'ids': np.arange(4, dtype=np.uint64),
            }
        )


def test_a_record_naming_another_dtype_does_not_rescue_a_uint64_source() -> None:
    """The refusal reads the value's own width, so no record over it makes a uint64 array admissible."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={
                'xyz': np.zeros((4, 3), dtype=np.float32),
                'ids': np.arange(4, dtype=np.uint64),
            },
            meta_data={'ids': {'dtype': 'int64', 'layout': ('ids',)}},
        )


def test_an_overwrite_leaves_the_meta_data_untouched() -> None:
    """A user may modify a field, but the meta data stays exactly what entered with it."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'intensity': np.arange(4, dtype=np.uint16),
        }
    )

    pc.intensity = torch.arange(4, dtype=torch.int64)

    assert pc.meta_data['intensity']['dtype'] == 'uint16'
    assert pc.intensity.dtype == torch.int64


def test_replacing_rgb_with_a_clone_preserves_its_colour_convention() -> None:
    """A clone carries the same dtype, and the dtype is what declares the convention, so the colour a display reads off the field is unchanged by the replacement."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'rgb': np.zeros((4, 3), dtype=np.uint16),
        }
    )

    pc.rgb = pc.rgb.clone()

    assert pc.meta_data['rgb']['dtype'] == 'uint16'


def test_a_float_rgb_outside_zero_to_one_is_refused_on_a_later_assignment() -> None:
    """The colour range is enforced on every assignment, not only at the door, so a field cannot be walked out of its own convention after it enters."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'rgb': np.full((4, 3), 0.5, dtype=np.float32),
        }
    )

    with pytest.raises(AssertionError):
        pc.rgb = torch.full((4, 3), 255.0, dtype=torch.float32)


def test_a_field_assigned_after_construction_is_named_by_no_meta_data() -> None:
    """The meta data is created once and never again, so a field arriving by attribute assignment sits outside it and carries only the dtype it means now."""
    pc = PointCloud(xyz=torch.zeros(4, 3, dtype=torch.float32))

    pc.feat = torch.zeros(4, 2, dtype=torch.float64)

    assert 'feat' not in pc.meta_data
    # its tensor is float64, which is what a field outside the meta data means
    assert pc.feat.dtype == torch.float64

    # the record is a plain mapping, so a name it does not hold is missing rather than refused
    with pytest.raises(KeyError):
        _ = pc.meta_data['feat']


def test_a_meta_data_handed_over_says_what_a_field_means() -> None:
    """The meta data is the only thing that can say an int32 tensor holds a uint16 colour, so a cloud built from another obj's stored fields is handed it whole rather than deriving it again from the tensors."""
    pc = PointCloud(
        data={
            'xyz': torch.zeros(4, 3, dtype=torch.float32),
            'intensity': torch.zeros(4, dtype=torch.int32),
        },
        meta_data={
            'xyz': {'dtype': 'float32', 'layout': ('xyz',)},
            'intensity': {'dtype': 'uint16', 'layout': ('intensity',)},
        },
    )

    assert pc.meta_data['intensity']['dtype'] == 'uint16'
    assert pc.meta_data['intensity']['layout'] == ('intensity',)
    # reading that entry gives 'uint16', not the int32 its tensor carries
    assert pc.meta_data['intensity']['dtype'] != 'int32'


def test_a_meta_data_handed_over_may_name_neither_more_nor_fewer_fields() -> None:
    """The meta data arrives whole rather than per field, so this door takes one naming a field that has since left beside one omitting a field that has since arrived."""
    pc = PointCloud(
        data={
            'xyz': torch.zeros(4, 3, dtype=torch.float32),
            'feat': torch.zeros(4, dtype=torch.float32),
        },
        meta_data={
            'xyz': {'dtype': 'float32', 'layout': ('xyz',)},
            'departed': {'dtype': 'float32', 'layout': ('departed',)},
        },
    )

    assert 'departed' in pc.meta_data
    assert 'feat' not in pc.meta_data
    # feat is read off its own tensor as float32, because no meta data entry names it
    assert pc.feat.dtype == torch.float32


def test_a_payload_without_the_meta_data_slot_is_refused() -> None:
    """A pickle written before the meta datas existed carries no such slot, and is refused so it is regenerated rather than restored into a cloud whose provenance is silently empty."""
    state = {
        '_fields': {'xyz': torch.zeros(4, 3, dtype=torch.float32)},
        '_length': 4,
        '_device': torch.device('cpu'),
    }

    with pytest.raises(AssertionError):
        PointCloud.__new__(PointCloud).__setstate__(state)


def test_a_point_cloud_survives_a_pickle_round_trip() -> None:
    """The meta data travels through pickle with the fields, so a cloud crossing a process boundary is the same cloud on the other side."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'intensity': np.arange(4, dtype=np.uint16),
        }
    )

    restored = pickle.loads(pickle.dumps(pc))

    assert restored.meta_data['intensity']['dtype'] == 'uint16'
    assert restored.meta_data['intensity']['layout'] == ('intensity',)
    assert restored.field_names() == ('xyz', 'intensity')


def test_coordinates_lead_whatever_order_the_fields_arrive_in() -> None:
    """Coordinates-first is insertion order now, so the constructor enters them first rather than trusting the caller's dict order."""
    pc = PointCloud(
        data={
            'feat': torch.zeros(4, dtype=torch.float32),
            'xyz': torch.zeros(4, 3, dtype=torch.float32),
        }
    )

    assert pc.field_names() == ('xyz', 'feat')


def test_a_layout_repeating_a_column_is_refused_in_a_handed_over_meta_data() -> None:
    """A column assembled into one field twice is not a layout, and the door refuses the record rather than reading past it."""
    with pytest.raises(AssertionError):
        PointCloud(
            data={'xyz': torch.zeros(4, 3, dtype=torch.float32)},
            meta_data={'xyz': {'dtype': 'float32', 'layout': ('a', 'a')}},
        )


def test_applying_the_same_meta_data_twice_changes_nothing() -> None:
    """Every load applies once inside the construction and again on the way out, and every save applies once more, so a second application that moved anything would move it on every ordinary path."""
    pc = PointCloud(
        data={
            'x': torch.zeros(4, dtype=torch.float32),
            'y': torch.ones(4, dtype=torch.float32),
            'z': torch.full((4,), 2.0, dtype=torch.float32),
            'intensity': np.arange(4, dtype=np.uint16),
        }
    )
    first_meta_data = {name: dict(entry) for name, entry in pc.meta_data.items()}
    first_fields = {name: getattr(pc, name).clone() for name in pc.field_names()}

    pc.apply_meta_data()

    assert pc.meta_data['xyz']['layout'] == ('x', 'y', 'z')
    assert pc.meta_data == first_meta_data
    assert pc.field_names() == tuple(first_fields.keys())
    for name, value in first_fields.items():
        assert torch.equal(getattr(pc, name), value)


def test_a_field_assembled_from_columns_can_be_split_back_into_them() -> None:
    """A later application names the source columns back out of the field the record already assembled, which is what makes the same columns reachable however they were last grouped."""
    pc = PointCloud(
        data={
            'x': torch.zeros(4, dtype=torch.float32),
            'y': torch.ones(4, dtype=torch.float32),
            'z': torch.full((4,), 2.0, dtype=torch.float32),
        }
    )

    # the coordinates are three columns wherever they are named xyz, so a regrouping onto two carries its own name
    pc.apply_meta_data(
        meta_data={'ground_plane': {'layout': ('x', 'y')}, 'z': {'layout': ('z',)}}
    )

    assert pc.ground_plane.shape[1] == 2
    assert pc.z.shape[0] == 4
    assert pc.meta_data['ground_plane']['layout'] == ('x', 'y')


def test_a_layout_override_enters_the_record_while_the_dtype_override_does_not() -> (
    None
):
    """The mapping's loaded side is what a caller asked for, and its dtype half stays what the source held whatever width the values were moved to."""
    pc = PointCloud(
        data={
            '0': torch.zeros(4, dtype=torch.float64),
            '1': torch.ones(4, dtype=torch.float64),
            '2': torch.full((4,), 2.0, dtype=torch.float64),
        },
        meta_data={'xyz': {'dtype': 'float32', 'layout': ('0', '1', '2')}},
    )

    assert pc.xyz.dtype == torch.float32
    assert pc.meta_data['xyz']['layout'] == ('0', '1', '2')
    assert pc.meta_data['xyz']['dtype'] == 'float64'


def test_applying_meta_data_hands_back_the_target_it_applied() -> None:
    """The dtype half a caller states never reaches the record, so the target is handed back for the writer that has to write the file at it."""
    pc = PointCloud(
        data={
            'intensity': np.arange(4, dtype=np.uint16),
            'xyz': np.zeros((4, 3), dtype=np.float32),
        }
    )

    target = pc.apply_meta_data(meta_data={'intensity': {'dtype': 'uint8'}})

    assert target['intensity']['dtype'] == 'uint8'
    assert pc.meta_data['intensity']['dtype'] == 'uint16'


def test_a_deleted_field_is_dropped_from_the_target_rather_than_aborting_it() -> None:
    """The record goes on naming a departed field, and the application that follows a deletion drops it instead of failing to find its columns."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((4, 3), dtype=np.float32),
            'intensity': np.arange(4, dtype=np.uint16),
        }
    )

    del pc.intensity

    target = pc.apply_meta_data()

    assert tuple(target.keys()) == ('xyz',)
