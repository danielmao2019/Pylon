"""A table is only right if it agrees with the system it describes, so each table is checked against what torch, numpy and the ply format actually carry rather than against itself."""

import numpy as np
import pytest
import torch

from utils.dtypes import (
    COLOR_RANGE,
    CONCEPTUAL_NAME,
    NUMPY_DTYPE,
    PLY_CHAR,
    TORCH_DTYPE,
    cast_lossless,
    conceptual_name_of,
    convert_color_convention,
)


def test_one_name_covers_both_systems_that_carry_the_dtype():
    """A numpy uint16 array and the uint16 array plyfile hands back for a u2 column name the same conceptual dtype, which is what lets one meta data entry describe either source."""
    assert (
        CONCEPTUAL_NAME[np.dtype('uint16')] == 'uint16'
    ), f"the numpy side of the table names a numpy uint16 array something else: CONCEPTUAL_NAME[np.dtype('uint16')]={CONCEPTUAL_NAME[np.dtype('uint16')]}"
    assert (
        CONCEPTUAL_NAME[torch.int32] == 'int32'
    ), f"the torch side of the table names a torch int32 tensor something else: CONCEPTUAL_NAME[torch.int32]={CONCEPTUAL_NAME[torch.int32]}"


def test_every_conceptual_name_has_a_torch_storage():
    """Every dtype a source can hand over must reach torch, since the obj stores torch tensors and nothing else."""
    for name in CONCEPTUAL_NAME.values():
        assert (
            name in TORCH_DTYPE
        ), f"a conceptual dtype name reaches no torch storage: name={name}, names TORCH_DTYPE carries={sorted(TORCH_DTYPE.keys())}"


def test_a_dtype_torch_carries_is_stored_as_itself():
    """Nothing is patched for a dtype torch already has, so the table is the identity everywhere the mismatch is not."""
    for name in (
        'int8',
        'uint8',
        'int16',
        'int32',
        'int64',
        'bool',
        'float16',
        'bfloat16',
        'float32',
        'float64',
    ):
        assert (
            CONCEPTUAL_NAME[TORCH_DTYPE[name]] == name
        ), f"a dtype torch carries is stored as another dtype: name={name}, TORCH_DTYPE[name]={TORCH_DTYPE[name]}, CONCEPTUAL_NAME[TORCH_DTYPE[name]]={CONCEPTUAL_NAME[TORCH_DTYPE[name]]}"


def test_the_widths_torch_lacks_are_stored_in_the_narrowest_width_that_holds_them():
    """uint16 goes to int32 and uint32 to int64 because those are the narrowest torch widths whose values cover them, and float128 goes to float64 because no torch float covers it."""
    assert (
        TORCH_DTYPE['uint16'] == torch.int32
    ), f"uint16 is not stored in the narrowest torch width holding it: TORCH_DTYPE['uint16']={TORCH_DTYPE['uint16']}"
    assert (
        TORCH_DTYPE['uint32'] == torch.int64
    ), f"uint32 is not stored in the narrowest torch width holding it: TORCH_DTYPE['uint32']={TORCH_DTYPE['uint32']}"
    assert (
        TORCH_DTYPE['float128'] == torch.float64
    ), f"float128 is not stored in the widest torch float: TORCH_DTYPE['float128']={TORCH_DTYPE['float128']}"


def test_the_numpy_table_spells_the_torch_storage():
    """A numpy source crosses into torch through this table, so an entry naming a width torch cannot hold would make the crossing fail."""
    for name, dtype in NUMPY_DTYPE.items():
        crossed = torch.from_numpy(np.empty((0,), dtype=dtype))
        assert (
            crossed.dtype == TORCH_DTYPE[name]
        ), f"the numpy spelling of a name crosses into a torch dtype the torch table does not name: name={name}, NUMPY_DTYPE[name]={dtype}, crossed.dtype={crossed.dtype}, TORCH_DTYPE[name]={TORCH_DTYPE[name]}"


def test_bfloat16_has_no_numpy_spelling():
    """numpy carries no bfloat16 at all, which is why the two crossings branch on membership in this table rather than assuming an entry."""
    assert (
        'bfloat16' not in NUMPY_DTYPE
    ), f"bfloat16 is spelled in numpy, which carries no such width: NUMPY_DTYPE['bfloat16']={NUMPY_DTYPE['bfloat16']}"


def test_every_conceptual_name_has_a_ply_column():
    """Save writes whatever the obj holds, so a name with no column would abort a save the design says succeeds."""
    for name in CONCEPTUAL_NAME.values():
        assert (
            name in PLY_CHAR
        ), f"a conceptual dtype name reaches no ply column: name={name}, names PLY_CHAR carries={sorted(PLY_CHAR.keys())}"


def test_the_ply_characters_are_the_eight_the_format_declares():
    """The format declares only char, uchar, short, ushort, int, uint, float and double, so a table emitting anything else would write a column no reader accepts."""
    for name, char in PLY_CHAR.items():
        assert char in (
            'i1',
            'u1',
            'i2',
            'u2',
            'i4',
            'u4',
            'f4',
            'f8',
        ), f"a ply column is written in a character the format does not declare: name={name}, PLY_CHAR[name]={char}"


def test_the_widths_ply_lacks_are_stored_in_the_widest_narrower_column():
    """ply has no 64-bit integer, no boolean and no half float, so each goes to the column the format does declare and the values decide whether the write survives."""
    assert (
        PLY_CHAR['int64'] == 'i4'
    ), f"int64 is not written in the widest signed column ply declares: PLY_CHAR['int64']={PLY_CHAR['int64']}"
    assert (
        PLY_CHAR['uint64'] == 'u4'
    ), f"uint64 is not written in the widest unsigned column ply declares: PLY_CHAR['uint64']={PLY_CHAR['uint64']}"
    assert (
        PLY_CHAR['bool'] == 'u1'
    ), f"bool is not written in the narrowest unsigned column ply declares: PLY_CHAR['bool']={PLY_CHAR['bool']}"
    assert (
        PLY_CHAR['float16'] == 'f4'
    ), f"float16 is not written in the narrowest float column ply declares: PLY_CHAR['float16']={PLY_CHAR['float16']}"
    assert (
        PLY_CHAR['bfloat16'] == 'f4'
    ), f"bfloat16 is not written in the narrowest float column ply declares: PLY_CHAR['bfloat16']={PLY_CHAR['bfloat16']}"
    assert (
        PLY_CHAR['float128'] == 'f8'
    ), f"float128 is not written in the widest float column ply declares: PLY_CHAR['float128']={PLY_CHAR['float128']}"


def test_the_colour_conventions_are_the_four_the_design_names():
    """A convention is told apart by dtype alone, and exactly four dtypes name one."""
    assert COLOR_RANGE['uint8'] == (
        0,
        255,
    ), f"the uint8 colour convention is not bounded by 0 and 255: COLOR_RANGE['uint8']={COLOR_RANGE['uint8']}"
    assert COLOR_RANGE['int8'] == (
        -128,
        127,
    ), f"the int8 colour convention is not bounded by -128 and 127: COLOR_RANGE['int8']={COLOR_RANGE['int8']}"
    assert COLOR_RANGE['uint16'] == (
        0,
        65535,
    ), f"the uint16 colour convention is not bounded by 0 and 65535: COLOR_RANGE['uint16']={COLOR_RANGE['uint16']}"
    assert COLOR_RANGE['float32'] == (0.0, 1.0) and COLOR_RANGE['float64'] == (
        0.0,
        1.0,
    ), f"a float colour convention is not bounded by 0.0 and 1.0: COLOR_RANGE['float32']={COLOR_RANGE['float32']}, COLOR_RANGE['float64']={COLOR_RANGE['float64']}"


def test_a_dtype_naming_no_convention_is_absent_from_the_table():
    """A bool or int32 colour has no convention to be read on, and its absence here is what refuses it rather than a branch somewhere else."""
    assert (
        'bool' not in COLOR_RANGE
    ), f"bool names a colour convention it has no bounds for: COLOR_RANGE['bool']={COLOR_RANGE['bool']}"
    assert (
        'int32' not in COLOR_RANGE
    ), f"int32 names a colour convention it has no bounds for: COLOR_RANGE['int32']={COLOR_RANGE['int32']}"


def test_a_recorded_dtype_is_what_its_own_storage_means():
    """A uint16 colour parked in an int32 tensor means uint16, an int32 tensor nothing recorded means int32, and data an override brought onto another storage means the storage it is held at."""
    assert (
        conceptual_name_of(torch.int32, 'uint16') == 'uint16'
    ), f"a uint16 colour still parked in its int32 storage is named something other than uint16: conceptual_name_of(torch.int32, 'uint16')={conceptual_name_of(torch.int32, 'uint16')}"
    assert (
        conceptual_name_of(torch.int32, None) == 'int32'
    ), f"an int32 tensor nothing recorded is named something other than int32: conceptual_name_of(torch.int32, None)={conceptual_name_of(torch.int32, None)}"
    assert (
        conceptual_name_of(torch.uint8, 'uint16') == 'uint8'
    ), f"data brought onto uint8 storage since it was recorded as uint16 is named something other than uint8: conceptual_name_of(torch.uint8, 'uint16')={conceptual_name_of(torch.uint8, 'uint16')}"


def test_a_lossless_cast_hands_back_the_target_dtype():
    """A cast whose values survive produces the named dtype carrying the same values."""
    values = np.array([1, 2, 3], dtype=np.int64)
    cast = cast_lossless(values, np.dtype('int32'))
    assert cast.dtype == np.dtype(
        'int32'
    ), f"a lossless cast hands back a dtype other than the one it was given: cast.dtype={cast.dtype}"
    assert np.array_equal(
        cast, values
    ), f"a lossless cast hands back values other than the ones it was given: cast={cast}, values={values}"


def test_a_cast_past_the_target_bounds_aborts():
    """A value the target cannot hold at all comes back a different value, and the round trip is what sees it."""
    values = np.array([2**40], dtype=np.int64)
    with pytest.raises(AssertionError):
        cast_lossless(values, np.dtype('int32'))


def test_a_cast_off_the_target_grid_aborts():
    """A value inside the target's bounds that the target's grid cannot land on is lost just as surely, which is why the test is a round trip and not a bounds check."""
    values = np.array([1.0 + 1e-12], dtype=np.float64)
    with pytest.raises(AssertionError):
        cast_lossless(values, np.dtype('float32'))


def test_a_conversion_maps_the_bounds_onto_the_bounds():
    """The two ends of a convention are the two ends of the one it maps onto, which is what makes the mapping a range mapping rather than a scale factor."""
    converted = convert_color_convention(
        values=np.array([0, 255], dtype=np.uint8),
        source_dtype='uint8',
        target_dtype='float32',
    )
    assert np.array_equal(
        converted, np.array([0.0, 1.0])
    ), f"the bounds of the uint8 convention do not map onto the bounds of the float32 convention: converted={converted}"


def test_a_signed_convention_maps_off_its_own_low_bound():
    """int8's range starts below zero, so the shift is the low bound rather than nothing."""
    converted = convert_color_convention(
        values=np.array([-128, 127], dtype=np.int8),
        source_dtype='int8',
        target_dtype='uint8',
    )
    assert np.array_equal(
        converted, np.array([0.0, 255.0])
    ), f"the bounds of the int8 convention do not map onto the bounds of the uint8 convention: converted={converted}"


def test_an_integer_target_lands_on_its_own_grid():
    """A target naming an integer convention rounds, since a colour between two of its steps is not a colour it can hold."""
    converted = convert_color_convention(
        values=np.array([1], dtype=np.uint16),
        source_dtype='uint16',
        target_dtype='uint8',
    )
    assert np.array_equal(
        converted, np.array([0.0])
    ), f"a colour mapped onto an integer convention did not land on that convention's grid: converted={converted}"


def test_a_float_target_keeps_what_the_mapping_gave_it():
    """A float convention has no grid to land on, so nothing is rounded away."""
    converted = convert_color_convention(
        values=np.array([1], dtype=np.uint8),
        source_dtype='uint8',
        target_dtype='float64',
    )
    assert np.array_equal(
        converted, np.array([1.0 / 255.0])
    ), f"a colour mapped onto a float convention was rounded away: converted={converted}, expected={1.0 / 255.0}"


def test_a_conversion_the_target_can_hold_comes_back_unchanged():
    """Losslessness is the round trip, and a source value sitting on the target's grid survives it, which is the case a save is allowed to write."""
    converted = convert_color_convention(
        values=np.array([257], dtype=np.uint16),
        source_dtype='uint16',
        target_dtype='uint8',
    )
    recovered = convert_color_convention(
        values=converted, source_dtype='uint8', target_dtype='uint16'
    )
    assert np.array_equal(
        recovered, np.array([257.0])
    ), f"a colour sitting on the target's grid did not survive the trip back: converted={converted}, recovered={recovered}"


def test_a_dtype_naming_no_convention_is_refused_by_the_conversion():
    """An int32 array is a colour only because a record says which convention it means, so the conversion refuses the dtype rather than inventing bounds for it."""
    with pytest.raises(AssertionError):
        convert_color_convention(
            values=np.array([1], dtype=np.int32),
            source_dtype='int32',
            target_dtype='uint8',
        )
