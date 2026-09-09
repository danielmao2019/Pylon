# utils/dtypes — code structure

## Code structure trees

`utils/dtypes.py`

```text
dtypes.py
├── # The conceptual dtype universe and the dtype each system stores a conceptual dtype in, as tables rather than as a containment rule: the set of dtypes is closed and small, so what a containment walk would compute is written down once here.
├── from typing import Union
├── import numpy as np
├── import torch
├── CONCEPTUAL_NAME  # np.dtype or torch.dtype -> conceptual dtype name; one table over both systems, since a numpy uint16 array and a ply u2 column plyfile hands back as uint16 name one dtype
├── TORCH_DTYPE  # conceptual dtype name -> the torch.dtype torch stores it in: itself where torch carries it, torch.int32 for uint16, torch.int64 for uint32, torch.float64 for float128
├── NUMPY_DTYPE  # conceptual dtype name -> the np.dtype spelling the same torch storage, which is the dtype a numpy source is cast to before it crosses into torch; bfloat16 has no entry, numpy carrying no such width
├── PLY_CHAR  # conceptual dtype name -> the ply dtype character its column is stored as: i1, u1, i2, u2, i4, u4, f4, f8, with int64 to i4, uint64 to u4, bool to u1, float16 and bfloat16 to f4 and float128 to f8
├── COLOR_RANGE  # conceptual dtype name -> the low and high bound of the colour convention that dtype names: uint8 0 to 255, int8 -128 to 127, uint16 0 to 65535, every float 0.0 to 1.0; a dtype absent here names no convention and a colour of it is refused
├── def cast_lossless(values: Union[np.ndarray, torch.Tensor], dtype: Union[np.dtype, torch.dtype]) -> Union[np.ndarray, torch.Tensor]
│   ├── # Casts values to a dtype and aborts rather than handing back values the cast changed, which is the promise every cast this module's callers make.
│   ├── impls cast = values cast to dtype
│   ├── impls recovered = cast cast back to the dtype values arrived in
│   ├── assert recovered equals values at every entry  # the round trip is the whole test: a value past the target's bounds and a value the target's grid cannot land on both come back different
│   └── return cast
└── def convert_color_convention(values: Union[np.ndarray, torch.Tensor], source_dtype: str, target_dtype: str) -> Union[np.ndarray, torch.Tensor]
    ├── # Maps colours off the range one conceptual dtype names onto the range another names, which is the one operation every reader of a colour at its own range performs.
    ├── assert source_dtype sits in COLOR_RANGE  # a dtype naming no convention has no bounds to map from, and its absence from the table is what refuses it
    ├── assert target_dtype sits in COLOR_RANGE
    ├── impls source_low, source_high = COLOR_RANGE[source_dtype]
    ├── impls target_low, target_high = COLOR_RANGE[target_dtype]
    ├── impls converted = values in double precision, shifted off source_low, scaled by the ratio of the two spans, and shifted onto target_low
    ├── if COLOR_RANGE[target_dtype] bounds an integer convention
    │   └── impls converted = converted rounded to the nearest integer
    └── return converted  # in double precision whichever convention it landed on and in the system its values arrived in, so a caller wanting the target's own storage casts it and a caller checking the trip back keeps the exact value
```
