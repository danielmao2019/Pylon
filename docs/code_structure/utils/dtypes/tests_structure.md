# utils/dtypes — tests structure

## Tests implementation structure

`tests/utils/test_dtypes.py`

```text
test_dtypes.py
├── # A table is only right if it agrees with the system it describes, so each table is checked against what torch, numpy and the ply format actually carry rather than against itself.
├── import numpy as np
├── import pytest
├── import torch
├── from utils.dtypes import COLOR_RANGE, CONCEPTUAL_NAME, NUMPY_DTYPE, PLY_CHAR, TORCH_DTYPE, cast_lossless, convert_color_convention
├── def test_one_name_covers_both_systems_that_carry_the_dtype()
│   ├── # A numpy uint16 array and the uint16 array plyfile hands back for a u2 column name the same conceptual dtype, which is what lets one meta data entry describe either source.
│   ├── assert CONCEPTUAL_NAME[np.dtype('uint16')] is 'uint16'
│   └── assert CONCEPTUAL_NAME[torch.int32] is 'int32'
├── def test_every_conceptual_name_has_a_torch_storage()
│   ├── # Every dtype a source can hand over must reach torch, since the obj stores torch tensors and nothing else.
│   └── for each name in CONCEPTUAL_NAME.values()
│       └── assert name sits in TORCH_DTYPE
├── def test_a_dtype_torch_carries_is_stored_as_itself()
│   ├── # Nothing is patched for a dtype torch already has, so the table is the identity everywhere the mismatch is not.
│   └── for each of 'int8', 'uint8', 'int16', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32' and 'float64'
│       └── assert CONCEPTUAL_NAME[TORCH_DTYPE[that name]] is that name
├── def test_the_widths_torch_lacks_are_stored_in_the_narrowest_width_that_holds_them()
│   ├── # uint16 goes to int32 and uint32 to int64 because those are the narrowest torch widths whose values cover them, and float128 goes to float64 because no torch float covers it.
│   ├── assert TORCH_DTYPE['uint16'] is torch.int32
│   ├── assert TORCH_DTYPE['uint32'] is torch.int64
│   └── assert TORCH_DTYPE['float128'] is torch.float64
├── def test_the_numpy_table_spells_the_torch_storage()
│   ├── # A numpy source crosses into torch through this table, so an entry naming a width torch cannot hold would make the crossing fail.
│   └── for each name, dtype in NUMPY_DTYPE
│       └── assert an empty numpy array of dtype is accepted by torch.from_numpy and lands in TORCH_DTYPE[name]
├── def test_bfloat16_has_no_numpy_spelling()
│   ├── # numpy carries no bfloat16 at all, which is why the two crossings branch on membership in this table rather than assuming an entry.
│   └── assert 'bfloat16' is absent from NUMPY_DTYPE
├── def test_every_conceptual_name_has_a_ply_column()
│   ├── # Save writes whatever the obj holds, so a name with no column would abort a save the design says succeeds.
│   └── for each name in CONCEPTUAL_NAME.values()
│       └── assert name sits in PLY_CHAR
├── def test_the_ply_characters_are_the_eight_the_format_declares()
│   ├── # The format declares only char, uchar, short, ushort, int, uint, float and double, so a table emitting anything else would write a column no reader accepts.
│   └── assert every value of PLY_CHAR sits in ('i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'f4', 'f8')
├── def test_the_widths_ply_lacks_are_stored_in_the_widest_narrower_column()
│   ├── # ply has no 64-bit integer, no boolean and no half float, so each goes to the column the format does declare and the values decide whether the write survives.
│   ├── assert PLY_CHAR['int64'] is 'i4'
│   ├── assert PLY_CHAR['uint64'] is 'u4'
│   ├── assert PLY_CHAR['bool'] is 'u1'
│   ├── assert PLY_CHAR['float16'] is 'f4'
│   ├── assert PLY_CHAR['bfloat16'] is 'f4'
│   └── assert PLY_CHAR['float128'] is 'f8'
├── def test_the_colour_conventions_are_the_four_the_design_names()
│   ├── # A convention is told apart by dtype alone, and exactly four dtypes name one.
│   ├── assert COLOR_RANGE['uint8'] is the bounds 0 and 255
│   ├── assert COLOR_RANGE['int8'] is the bounds -128 and 127
│   ├── assert COLOR_RANGE['uint16'] is the bounds 0 and 65535
│   └── assert COLOR_RANGE['float32'] and COLOR_RANGE['float64'] are both the bounds 0.0 and 1.0
├── def test_a_dtype_naming_no_convention_is_absent_from_the_table()
│   ├── # A bool or int32 colour has no convention to be read on, and its absence here is what refuses it rather than a branch somewhere else.
│   ├── assert 'bool' is absent from COLOR_RANGE
│   └── assert 'int32' is absent from COLOR_RANGE
├── def test_a_lossless_cast_hands_back_the_target_dtype()
│   ├── # A cast whose values survive produces the named dtype carrying the same values.
│   ├── calls cast_lossless(an int64 array of small values, np.dtype('int32'))
│   ├── assert the result is int32
│   └── assert the result carries the values it was given
├── def test_a_cast_past_the_target_bounds_aborts()
│   ├── # A value the target cannot hold at all comes back a different value, and the round trip is what sees it.
│   └── with pytest.raises(AssertionError)
│       └── calls cast_lossless(an int64 array carrying a value beyond int32, np.dtype('int32'))
├── def test_a_cast_off_the_target_grid_aborts()
│   ├── # A value inside the target's bounds that the target's grid cannot land on is lost just as surely, which is why the test is a round trip and not a bounds check.
│   └── with pytest.raises(AssertionError)
│       └── calls cast_lossless(a float64 array carrying a value float32 rounds, np.dtype('float32'))
├── def test_a_conversion_maps_the_bounds_onto_the_bounds()
│   ├── # The two ends of a convention are the two ends of the one it maps onto, which is what makes the mapping a range mapping rather than a scale factor.
│   ├── calls convert_color_convention(values=a uint8 array holding 0 and 255, source_dtype='uint8', target_dtype='float32')
│   └── assert what it mapped holds 0.0 and 1.0
├── def test_a_signed_convention_maps_off_its_own_low_bound()
│   ├── # int8's range starts below zero, so the shift is the low bound rather than nothing.
│   ├── calls convert_color_convention(values=an int8 array holding -128 and 127, source_dtype='int8', target_dtype='uint8')
│   └── assert what it mapped holds 0 and 255
├── def test_an_integer_target_lands_on_its_own_grid()
│   ├── # A target naming an integer convention rounds, since a colour between two of its steps is not a colour it can hold.
│   ├── calls convert_color_convention(values=a uint16 array holding 1, source_dtype='uint16', target_dtype='uint8')
│   └── assert what it mapped holds 0
├── def test_a_float_target_keeps_what_the_mapping_gave_it()
│   ├── # A float convention has no grid to land on, so nothing is rounded away.
│   ├── calls convert_color_convention(values=a uint8 array holding 1, source_dtype='uint8', target_dtype='float64')
│   └── assert what it mapped holds one 255th rather than zero
├── def test_a_conversion_the_target_can_hold_comes_back_unchanged()
│   ├── # Losslessness is the round trip, and a source value sitting on the target's grid survives it, which is the case a save is allowed to write.
│   ├── calls convert_color_convention(values=a uint16 array holding 257, source_dtype='uint16', target_dtype='uint8')
│   ├── calls convert_color_convention(values=what it mapped, source_dtype='uint8', target_dtype='uint16')
│   └── assert what it mapped back holds 257
└── def test_a_dtype_naming_no_convention_is_refused_by_the_conversion()
    ├── # An int32 array is a colour only because a record says which convention it means, so the conversion refuses the dtype rather than inventing bounds for it.
    └── with pytest.raises(AssertionError)
        └── calls convert_color_convention(values=an int32 array, source_dtype='int32', target_dtype='uint8')
```
