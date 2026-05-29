"""Generic glTF/GLB I/O: the chunked file, its typed accessor arrays, and embedded image bytes.

This module reads and writes glTF/GLB data at the byte/array level only. It
parses and serializes the GLB chunked container, decodes and builds typed
accessor arrays, and slices out and embeds raw image bytes. It does not codec
image pixels (that is utils.io.image) and does not interpret glTF semantics
(which primitive/material/accessor means what is the caller's concern).
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

_GLB_MAGIC: int = 0x46546C67  # b"glTF"
_GLB_VERSION: int = 2
_GLB_HEADER_SIZE: int = 12
_GLB_CHUNK_HEADER_SIZE: int = 8
_GLB_JSON_CHUNK_TYPE: int = 0x4E4F534A  # b"JSON"
_GLB_BIN_CHUNK_TYPE: int = 0x004E4942  # b"BIN\0"

_COMPONENT_DTYPES: Dict[int, np.dtype] = {
    5120: np.dtype("int8"),
    5121: np.dtype("uint8"),
    5122: np.dtype("int16"),
    5123: np.dtype("uint16"),
    5125: np.dtype("uint32"),
    5126: np.dtype("float32"),
}

_COMPONENT_COUNTS: Dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

_NUMPY_COMPONENT_TYPES: Dict[str, int] = {
    np.dtype("int8").name: 5120,
    np.dtype("uint8").name: 5121,
    np.dtype("int16").name: 5122,
    np.dtype("uint16").name: 5123,
    np.dtype("uint32").name: 5125,
    np.dtype("float32").name: 5126,
}

_ACCESSOR_TYPES: Dict[int, str] = {
    1: "SCALAR",
    2: "VEC2",
    3: "VEC3",
    4: "VEC4",
}


def load_glb_json_and_bin(path: Union[str, Path]) -> Tuple[Dict[str, Any], bytes]:
    """Parse a GLB container into its glTF JSON document and binary buffer blob.

    Args:
        path: Filesystem path to the GLB file.

    Returns:
        Tuple ``(gltf, binary_blob)`` where ``gltf`` is the decoded glTF JSON
        dictionary and ``binary_blob`` is the concatenated GLB BIN chunk bytes.
    """
    assert isinstance(
        path, (str, Path)
    ), f"path must be str or Path, got type={type(path).__name__}"
    with open(file=str(path), mode="rb") as stream:
        header = stream.read(_GLB_HEADER_SIZE)
        magic, version, total_length = struct.unpack("<III", header)
        assert (
            magic == _GLB_MAGIC
        ), f"GLB magic mismatch, expected glTF, got magic={hex(magic)} for path={str(path)!r}"
        assert (
            version == _GLB_VERSION
        ), f"only GLB version 2 supported, got version={version} for path={str(path)!r}"
        json_chunk_length, json_chunk_type = struct.unpack(
            "<II", stream.read(_GLB_CHUNK_HEADER_SIZE)
        )
        assert (
            json_chunk_type == _GLB_JSON_CHUNK_TYPE
        ), f"first GLB chunk must be JSON, got chunk_type={hex(json_chunk_type)} for path={str(path)!r}"
        json_bytes = stream.read(json_chunk_length)
        gltf = json.loads(json_bytes.decode("utf-8"))
        bin_chunk_length, bin_chunk_type = struct.unpack(
            "<II", stream.read(_GLB_CHUNK_HEADER_SIZE)
        )
        assert (
            bin_chunk_type == _GLB_BIN_CHUNK_TYPE
        ), f"second GLB chunk must be BIN, got chunk_type={hex(bin_chunk_type)} for path={str(path)!r}"
        binary_blob = stream.read(bin_chunk_length)
        assert (
            total_length
            == _GLB_HEADER_SIZE
            + _GLB_CHUNK_HEADER_SIZE
            + json_chunk_length
            + _GLB_CHUNK_HEADER_SIZE
            + bin_chunk_length
        ), f"GLB total length mismatch, got total_length={total_length} for path={str(path)!r}"
    return gltf, binary_blob


def read_accessor(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    accessor_index: int,
) -> np.ndarray:
    """Decode one glTF accessor (dense values plus any sparse overlay) into a numpy array.

    Handles both dense bufferView-backed accessors and GLB sparse accessors:
    the dense bufferView may be absent (all values implicitly zero) with the
    non-zero entries living in the accessor's ``sparse`` block.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        accessor_index: Accessor index into ``gltf["accessors"]``.

    Returns:
        ``np.ndarray`` of shape ``(count, component_count)`` (or ``(count,)``
        for ``SCALAR`` accessors), dtype matching the accessor component type.
    """
    assert isinstance(gltf, dict), f"gltf must be dict, got type={type(gltf).__name__}"
    assert isinstance(
        accessor_index, int
    ), f"accessor_index must be int, got type={type(accessor_index).__name__}"
    accessor = gltf["accessors"][accessor_index]
    component_dtype = _component_dtype(component_type=accessor["componentType"])
    component_count = _component_count(accessor_type=accessor["type"])
    count = int(accessor["count"])
    if "bufferView" in accessor:
        buffer_view_bytes = _read_buffer_view_bytes(
            gltf=gltf,
            binary_blob=binary_blob,
            buffer_view_index=accessor["bufferView"],
        )
        byte_offset = accessor.get("byteOffset", 0)
        element_byte_size = component_dtype.itemsize * component_count
        needed_bytes = byte_offset + element_byte_size * count
        assert needed_bytes <= len(buffer_view_bytes), (
            f"accessor exceeds buffer view, needed_bytes={needed_bytes}, "
            f"buffer_view_bytes={len(buffer_view_bytes)}, accessor_index={accessor_index}"
        )
        flat = np.frombuffer(
            buffer=buffer_view_bytes[
                byte_offset : byte_offset + element_byte_size * count
            ],
            dtype=component_dtype,
        )
        array = flat.reshape(count, component_count).copy()
    else:
        array = np.zeros((count, component_count), dtype=component_dtype)

    sparse = accessor.get("sparse")
    if isinstance(sparse, dict) and int(sparse.get("count", 0)) > 0:
        _apply_sparse_overlay(
            gltf=gltf,
            binary_blob=binary_blob,
            sparse=sparse,
            target_array=array,
        )

    if component_count == 1:
        return array.reshape(count).copy()
    return array


def read_image_bytes(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    image_index: int,
) -> bytes:
    """Extract the raw encoded bytes of one glTF image from its buffer view.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        image_index: Image index into ``gltf["images"]``.

    Returns:
        Raw embedded image bytes for the addressed image.
    """
    assert isinstance(gltf, dict), f"gltf must be dict, got type={type(gltf).__name__}"
    assert isinstance(
        binary_blob, (bytes, bytearray)
    ), f"binary_blob must be bytes-like, got type={type(binary_blob).__name__}"
    assert isinstance(
        image_index, int
    ), f"image_index must be int, got type={type(image_index).__name__}"
    image_def = gltf["images"][image_index]
    assert (
        "bufferView" in image_def
    ), f"only buffer-view embedded images are supported, got image_def_keys={list(image_def.keys())}"
    return _read_buffer_view_bytes(
        gltf=gltf,
        binary_blob=binary_blob,
        buffer_view_index=image_def["bufferView"],
    )


def write_glb(gltf: Dict[str, Any], binary_blob: bytes, path: Union[str, Path]) -> None:
    """Serialize a glTF JSON document + binary buffer into the GLB chunked container on disk.

    The single buffer's ``byteLength`` is set to the binary blob length, then a
    12-byte header, a 4-byte-padded JSON chunk, and a 4-byte-padded BIN chunk
    are written in order.

    Args:
        gltf: glTF JSON dictionary; ``gltf["buffers"][0]["byteLength"]`` is set here.
        binary_blob: The BIN chunk payload (concatenated bufferView bytes).
        path: Output filesystem path for the GLB file.

    Returns:
        None. Writes the GLB file at ``path``.
    """
    assert isinstance(gltf, dict), f"gltf must be dict, got type={type(gltf).__name__}"
    assert isinstance(
        binary_blob, (bytes, bytearray)
    ), f"binary_blob must be bytes-like, got type={type(binary_blob).__name__}"
    assert isinstance(
        path, (str, Path)
    ), f"path must be str or Path, got type={type(path).__name__}"
    bin_payload = bytes(binary_blob)
    if not gltf.get("buffers"):
        gltf["buffers"] = [{}]
    gltf["buffers"][0]["byteLength"] = len(bin_payload)
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = json_bytes + b" " * ((4 - (len(json_bytes) % 4)) % 4)
    bin_chunk = bin_payload + b"\x00" * ((4 - (len(bin_payload) % 4)) % 4)
    total_length = (
        _GLB_HEADER_SIZE
        + _GLB_CHUNK_HEADER_SIZE
        + len(json_chunk)
        + _GLB_CHUNK_HEADER_SIZE
        + len(bin_chunk)
    )
    with open(file=str(path), mode="wb") as stream:
        stream.write(struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length))
        stream.write(struct.pack("<II", len(json_chunk), _GLB_JSON_CHUNK_TYPE))
        stream.write(json_chunk)
        stream.write(struct.pack("<II", len(bin_chunk), _GLB_BIN_CHUNK_TYPE))
        stream.write(bin_chunk)


def append_accessor(
    gltf: Dict[str, Any],
    binary_blob: bytearray,
    array: np.ndarray,
    target: Optional[int],
) -> int:
    """Append an array to the buffer as a new bufferView + accessor, returning the accessor index.

    The accessor ``componentType`` and ``type`` are inferred from the array's
    dtype and shape; per-component ``min``/``max`` are recorded. A 1-D array is
    a ``SCALAR`` accessor; a 2-D array is typed by its second dimension.

    Args:
        gltf: glTF JSON dictionary; its ``bufferViews`` and ``accessors`` lists grow.
        binary_blob: Mutable BIN payload; the array bytes are appended (4-byte aligned).
        array: Source array, 1-D ``(count,)`` or 2-D ``(count, component_count)``.
        target: bufferView ``target`` (34962 ARRAY_BUFFER / 34963 ELEMENT_ARRAY_BUFFER), or None.

    Returns:
        Index of the newly appended accessor in ``gltf["accessors"]``.
    """
    assert isinstance(gltf, dict), f"gltf must be dict, got type={type(gltf).__name__}"
    assert isinstance(
        binary_blob, bytearray
    ), f"binary_blob must be bytearray, got type={type(binary_blob).__name__}"
    assert isinstance(
        array, np.ndarray
    ), f"array must be np.ndarray, got type={type(array).__name__}"
    assert target is None or isinstance(
        target, int
    ), f"target must be int or None, got type={type(target).__name__}"
    assert array.ndim in (1, 2), f"array must be 1-D or 2-D, got shape={array.shape}"
    contiguous = np.ascontiguousarray(array)
    count = int(contiguous.shape[0])
    component_count = 1 if contiguous.ndim == 1 else int(contiguous.shape[1])
    component_type = _numpy_component_type(dtype=contiguous.dtype)
    accessor_type = _accessor_type(num_components=component_count)

    binary_blob.extend(b"\x00" * ((4 - (len(binary_blob) % 4)) % 4))
    byte_offset = len(binary_blob)
    raw = contiguous.tobytes()
    binary_blob.extend(raw)

    buffer_view: Dict[str, Any] = {
        "buffer": 0,
        "byteOffset": byte_offset,
        "byteLength": len(raw),
    }
    if target is not None:
        buffer_view["target"] = target
    gltf.setdefault("bufferViews", []).append(buffer_view)

    flat = contiguous.reshape(count, component_count)
    accessor: Dict[str, Any] = {
        "bufferView": len(gltf["bufferViews"]) - 1,
        "componentType": component_type,
        "count": count,
        "type": accessor_type,
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
    }
    gltf.setdefault("accessors", []).append(accessor)
    return len(gltf["accessors"]) - 1


def append_image(
    gltf: Dict[str, Any],
    binary_blob: bytearray,
    image_bytes: bytes,
    mime_type: str,
) -> int:
    """Append encoded image bytes to the buffer as a new bufferView + image, returning the image index.

    Args:
        gltf: glTF JSON dictionary; its ``bufferViews`` and ``images`` lists grow.
        binary_blob: Mutable BIN payload; the image bytes are appended (4-byte aligned).
        image_bytes: Encoded image bytes (PNG / JPEG / ...).
        mime_type: MIME type for the image (e.g. ``"image/png"``).

    Returns:
        Index of the newly appended image in ``gltf["images"]``.
    """
    assert isinstance(gltf, dict), f"gltf must be dict, got type={type(gltf).__name__}"
    assert isinstance(
        binary_blob, bytearray
    ), f"binary_blob must be bytearray, got type={type(binary_blob).__name__}"
    assert isinstance(
        image_bytes, (bytes, bytearray)
    ), f"image_bytes must be bytes-like, got type={type(image_bytes).__name__}"
    assert isinstance(
        mime_type, str
    ), f"mime_type must be str, got type={type(mime_type).__name__}"
    binary_blob.extend(b"\x00" * ((4 - (len(binary_blob) % 4)) % 4))
    byte_offset = len(binary_blob)
    raw = bytes(image_bytes)
    binary_blob.extend(raw)

    buffer_view: Dict[str, Any] = {
        "buffer": 0,
        "byteOffset": byte_offset,
        "byteLength": len(raw),
    }
    gltf.setdefault("bufferViews", []).append(buffer_view)
    gltf.setdefault("images", []).append(
        {
            "bufferView": len(gltf["bufferViews"]) - 1,
            "mimeType": mime_type,
        }
    )
    return len(gltf["images"]) - 1


def _apply_sparse_overlay(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    sparse: Dict[str, Any],
    target_array: np.ndarray,
) -> None:
    """Overwrite the glTF sparse-accessor index/value pairs onto the densely-read accessor values in place.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        sparse: The accessor's ``"sparse"`` sub-dict (``"count"``, ``"indices"``, ``"values"``).
        target_array: Dense backing array ``(count, component_count)`` mutated in place.

    Returns:
        None. Mutates ``target_array``.
    """
    sparse_count: int = int(sparse["count"])
    index_dtype = _component_dtype(
        component_type=int(sparse["indices"]["componentType"])
    )
    index_bytes = _read_buffer_view_bytes(
        gltf=gltf,
        binary_blob=binary_blob,
        buffer_view_index=int(sparse["indices"]["bufferView"]),
    )
    index_byte_offset = int(sparse["indices"].get("byteOffset", 0))
    index_array = np.frombuffer(
        buffer=index_bytes[
            index_byte_offset : index_byte_offset + sparse_count * index_dtype.itemsize
        ],
        dtype=index_dtype,
    ).astype(np.int64)

    component_count = int(target_array.shape[1])
    value_component_dtype = target_array.dtype
    value_element_byte_size = value_component_dtype.itemsize * component_count
    value_bytes = _read_buffer_view_bytes(
        gltf=gltf,
        binary_blob=binary_blob,
        buffer_view_index=int(sparse["values"]["bufferView"]),
    )
    value_byte_offset = int(sparse["values"].get("byteOffset", 0))
    value_array = np.frombuffer(
        buffer=value_bytes[
            value_byte_offset : value_byte_offset
            + sparse_count * value_element_byte_size
        ],
        dtype=value_component_dtype,
    ).reshape(sparse_count, component_count)
    target_array[index_array] = value_array


def _read_buffer_view_bytes(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    buffer_view_index: int,
) -> bytes:
    """Slice the raw bytes of one glTF buffer view out of the binary blob.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        buffer_view_index: Buffer-view index into ``gltf["bufferViews"]``.

    Returns:
        Byte slice referenced by that buffer view.
    """
    assert isinstance(
        buffer_view_index, int
    ), f"buffer_view_index must be int, got type={type(buffer_view_index).__name__}"
    buffer_view = gltf["bufferViews"][buffer_view_index]
    byte_offset = buffer_view.get("byteOffset", 0)
    byte_length = buffer_view["byteLength"]
    assert byte_offset + byte_length <= len(binary_blob), (
        f"bufferView exceeds binary blob, byte_offset={byte_offset}, "
        f"byte_length={byte_length}, blob_size={len(binary_blob)}"
    )
    return bytes(binary_blob[byte_offset : byte_offset + byte_length])


def _component_dtype(component_type: int) -> np.dtype:
    """Map a glTF accessor componentType code to its numpy dtype.

    Args:
        component_type: glTF component-type code (``5120..5126`` per the spec).

    Returns:
        Matching ``np.dtype``.
    """
    assert (
        component_type in _COMPONENT_DTYPES
    ), f"unknown GLB component type, got component_type={component_type}"
    return _COMPONENT_DTYPES[component_type]


def _component_count(accessor_type: str) -> int:
    """Map a glTF accessor type string to its component count.

    Args:
        accessor_type: glTF accessor type (``SCALAR``, ``VEC2``, ``VEC3``, ``VEC4``, ``MAT2``, ``MAT3``, ``MAT4``).

    Returns:
        Number of scalar components per element.
    """
    assert (
        accessor_type in _COMPONENT_COUNTS
    ), f"unknown GLB accessor type, got accessor_type={accessor_type!r}"
    return _COMPONENT_COUNTS[accessor_type]


def _numpy_component_type(dtype: np.dtype) -> int:
    """Map a numpy dtype to its glTF accessor componentType code.

    Args:
        dtype: Source array ``np.dtype``.

    Returns:
        glTF component-type code (``5120..5126``).
    """
    assert (
        dtype.name in _NUMPY_COMPONENT_TYPES
    ), f"unsupported numpy dtype for glTF accessor, got dtype={dtype.name}"
    return _NUMPY_COMPONENT_TYPES[dtype.name]


def _accessor_type(num_components: int) -> str:
    """Map a component count to its glTF accessor type string.

    Args:
        num_components: Number of scalar components per element (1..4).

    Returns:
        glTF accessor type string (``SCALAR`` / ``VEC2`` / ``VEC3`` / ``VEC4``).
    """
    assert (
        num_components in _ACCESSOR_TYPES
    ), f"unsupported component count for glTF accessor type, got num_components={num_components}"
    return _ACCESSOR_TYPES[num_components]
