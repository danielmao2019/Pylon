"""The conceptual dtype universe and the dtype each system stores a conceptual dtype in, as tables rather than as a containment rule: the set of dtypes is closed and small, so what a containment walk would compute is written down once here."""

from typing import Union

import numpy as np
import torch

# np.dtype or torch.dtype -> conceptual dtype name; one table over both systems, since a numpy uint16 array and a ply u2 column plyfile hands back as uint16 name one dtype
CONCEPTUAL_NAME = {
    np.dtype('int8'): 'int8',
    np.dtype('uint8'): 'uint8',
    np.dtype('int16'): 'int16',
    np.dtype('uint16'): 'uint16',
    np.dtype('int32'): 'int32',
    np.dtype('uint32'): 'uint32',
    np.dtype('int64'): 'int64',
    np.dtype('uint64'): 'uint64',
    np.dtype('bool'): 'bool',
    np.dtype('float16'): 'float16',
    np.dtype('float32'): 'float32',
    np.dtype('float64'): 'float64',
    np.dtype('float128'): 'float128',
    np.dtype('complex64'): 'complex64',
    np.dtype('complex128'): 'complex128',
    np.dtype('complex256'): 'complex256',
    torch.int8: 'int8',
    torch.uint8: 'uint8',
    torch.int16: 'int16',
    torch.int32: 'int32',
    torch.int64: 'int64',
    torch.bool: 'bool',
    torch.float16: 'float16',
    torch.bfloat16: 'bfloat16',
    torch.float32: 'float32',
    torch.float64: 'float64',
    torch.complex64: 'complex64',
    torch.complex128: 'complex128',
}

# conceptual dtype name -> the torch.dtype torch stores it in: itself where torch carries it, torch.int32 for uint16, torch.int64 for uint32, torch.float64 for float128
TORCH_DTYPE = {
    'int8': torch.int8,
    'uint8': torch.uint8,
    'int16': torch.int16,
    'uint16': torch.int32,
    'int32': torch.int32,
    'uint32': torch.int64,
    'int64': torch.int64,
    'uint64': torch.int64,
    'bool': torch.bool,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float64': torch.float64,
    'float128': torch.float64,
    'complex64': torch.complex64,
    'complex128': torch.complex128,
    'complex256': torch.complex128,
}

# conceptual dtype name -> the np.dtype spelling the same torch storage, which is the dtype a numpy source is cast to before it crosses into torch; bfloat16 has no entry, numpy carrying no such width
NUMPY_DTYPE = {
    'int8': np.dtype('int8'),
    'uint8': np.dtype('uint8'),
    'int16': np.dtype('int16'),
    'uint16': np.dtype('int32'),
    'int32': np.dtype('int32'),
    'uint32': np.dtype('int64'),
    'int64': np.dtype('int64'),
    'uint64': np.dtype('int64'),
    'bool': np.dtype('bool'),
    'float16': np.dtype('float16'),
    'float32': np.dtype('float32'),
    'float64': np.dtype('float64'),
    'float128': np.dtype('float64'),
    'complex64': np.dtype('complex64'),
    'complex128': np.dtype('complex128'),
    'complex256': np.dtype('complex128'),
}

# conceptual dtype name -> the ply dtype character its column is stored as: i1, u1, i2, u2, i4, u4, f4, f8, with int64 to i4, uint64 to u4, bool to u1, float16 and bfloat16 to f4 and float128 to f8
PLY_CHAR = {
    'int8': 'i1',
    'uint8': 'u1',
    'int16': 'i2',
    'uint16': 'u2',
    'int32': 'i4',
    'uint32': 'u4',
    'int64': 'i4',
    'uint64': 'u4',
    'bool': 'u1',
    'float16': 'f4',
    'bfloat16': 'f4',
    'float32': 'f4',
    'float64': 'f8',
    'float128': 'f8',
    'complex64': 'f4',
    'complex128': 'f8',
    'complex256': 'f8',
}

# conceptual dtype name -> the low and high bound of the colour convention that dtype names: uint8 0 to 255, int8 -128 to 127, uint16 0 to 65535, every float 0.0 to 1.0; a dtype absent here names no convention and a colour of it is refused
COLOR_RANGE = {
    'uint8': (0, 255),
    'int8': (-128, 127),
    'uint16': (0, 65535),
    'float16': (0.0, 1.0),
    'bfloat16': (0.0, 1.0),
    'float32': (0.0, 1.0),
    'float64': (0.0, 1.0),
    'float128': (0.0, 1.0),
}


def cast_lossless(
    values: Union[np.ndarray, torch.Tensor], dtype: Union[np.dtype, torch.dtype]
) -> Union[np.ndarray, torch.Tensor]:
    """Casts values to a dtype and aborts rather than handing back values the cast changed, which is the promise every cast this module's callers make.

    Args:
        values: The values to cast, as a numpy array or a torch tensor of any shape, carrying any dtype this module names.
        dtype: The dtype to cast to, as an np.dtype naming numpy storage or a torch.dtype naming torch storage; a numpy source crossing into a torch.dtype is spelled through NUMPY_DTYPE before torch.from_numpy carries it over.

    Returns:
        The cast values, of the same shape as values, as a numpy array carrying dtype when dtype is an np.dtype and as a torch tensor carrying dtype when dtype is a torch.dtype.
    """
    if isinstance(values, np.ndarray) and isinstance(dtype, torch.dtype):
        cast = torch.from_numpy(values.astype(NUMPY_DTYPE[CONCEPTUAL_NAME[dtype]]))
    elif isinstance(values, np.ndarray):
        cast = values.astype(dtype)
    elif isinstance(dtype, torch.dtype):
        cast = values.to(dtype)
    else:
        cast = values.numpy().astype(dtype)

    if isinstance(values, torch.Tensor) and isinstance(cast, torch.Tensor):
        # both sides are read in torch, on the device they already sit on: the dtype torch promotes the pair to contains both of their sets, torch naming no uint64 for the promotion to leave a pair of integers with
        common = torch.promote_types(values.dtype, cast.dtype)
        compared = values.to(common)
        recompared = cast.to(common)
    else:
        # each side is read as numpy first, a bfloat16 tensor through float64 since numpy carries no such width and float64 holds every bfloat16 value exactly
        if isinstance(values, torch.Tensor) and values.dtype == torch.bfloat16:
            compared = values.to(torch.float64).cpu().numpy()
        elif isinstance(values, torch.Tensor):
            compared = values.cpu().numpy()
        else:
            compared = values

        if isinstance(cast, torch.Tensor) and cast.dtype == torch.bfloat16:
            recompared = cast.to(torch.float64).cpu().numpy()
        elif isinstance(cast, torch.Tensor):
            recompared = cast.cpu().numpy()
        else:
            recompared = cast

        # both sides are read where neither wraps nor rounds: the dtype numpy promotes the pair to, falling back to python's unbounded int where that promotion leaves the integers, as an int64 and uint64 pair does
        common = np.promote_types(compared.dtype, recompared.dtype)
        if (
            compared.dtype.kind in 'bui'
            and recompared.dtype.kind in 'bui'
            and common.kind not in 'bui'
        ):
            common = np.dtype(object)
        compared = compared.astype(common)
        recompared = recompared.astype(common)

    # the cast's own value against the value it came from is the whole test, and a NaN that stays NaN is not a value the cast changed
    matched = (recompared == compared) | (
        (recompared != recompared) & (compared != compared)
    )
    assert bool(
        matched.all()
    ), f"casting to dtype changed values: dtype={dtype}, values.dtype={values.dtype}, changed values={compared[~matched]}, cast to={recompared[~matched]}"

    return cast


def convert_color_convention(
    values: Union[np.ndarray, torch.Tensor], source_dtype: str, target_dtype: str
) -> Union[np.ndarray, torch.Tensor]:
    """Maps colours off the range one conceptual dtype names onto the range another names, which is the one operation every reader of a colour at its own range performs.

    Args:
        values: The colours to map, as a numpy array or a torch tensor of any shape, carrying values on the convention source_dtype names.
        source_dtype: The conceptual dtype name whose colour convention the values are read on, as a string keying COLOR_RANGE.
        target_dtype: The conceptual dtype name whose colour convention the values are mapped onto, as a string keying COLOR_RANGE.

    Returns:
        The mapped colours, of the same shape as values, in double precision (np.float64 for a numpy array, torch.float64 for a torch tensor) on the convention target_dtype names.
    """
    assert (
        source_dtype in COLOR_RANGE
    ), f"a dtype naming no colour convention has no bounds to map from: source_dtype={source_dtype}, dtypes naming a convention={sorted(COLOR_RANGE.keys())}"
    assert (
        target_dtype in COLOR_RANGE
    ), f"a dtype naming no colour convention has no bounds to map onto: target_dtype={target_dtype}, dtypes naming a convention={sorted(COLOR_RANGE.keys())}"

    source_low, source_high = COLOR_RANGE[source_dtype]
    target_low, target_high = COLOR_RANGE[target_dtype]
    converted = (
        values.astype(np.float64)
        if isinstance(values, np.ndarray)
        else values.to(torch.float64)
    )
    converted = target_low + (converted - source_low) * (
        (target_high - target_low) / (source_high - source_low)
    )
    if isinstance(target_low, int):
        converted = converted.round()

    return converted
