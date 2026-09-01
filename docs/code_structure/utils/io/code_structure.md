# Utils IO Code Structure

## 1. Code structure trees

`utils/io/glb.py`

```text
glb.py
├── import json
├── import struct
├── import numpy as np
├── _GLB_MAGIC              # int / 0x46546C67, b"glTF"
├── _GLB_VERSION            # int / 2, the GLB container version written
├── _GLB_HEADER_SIZE        # int / 12, the container header's own byte count
├── _GLB_CHUNK_HEADER_SIZE  # int / 8, one chunk header's byte count
├── _GLB_JSON_CHUNK_TYPE    # int / 0x4E4F534A, b"JSON"
├── _GLB_BIN_CHUNK_TYPE     # int / 0x004E4942, b"BIN\0"
├── def load_glb_json_and_bin(path: Union[str, Path]) -> Tuple[Dict[str, Any], bytes]
│   ├── # Parses a GLB container into its glTF JSON document and binary buffer blob.
│   ├── impls read the file's bytes
│   ├── impls assert the 12-byte header carries the glTF magic
│   ├── impls gltf = the JSON chunk's bytes parsed as the glTF document
│   ├── impls binary_blob = the BIN chunk's bytes
│   └── return gltf, binary_blob
├── def read_accessor(gltf: Dict[str, Any], binary_blob: bytes, accessor_index: int) -> np.ndarray
│   ├── # Decodes one glTF accessor (dense values plus any sparse overlay) into a numpy array.
│   ├── calls _read_buffer_view_bytes
│   ├── calls _component_dtype
│   ├── calls _component_count
│   └── calls _apply_sparse_overlay
├── def read_image_bytes(gltf: Dict[str, Any], binary_blob: bytes, image_index: int) -> bytes
│   ├── # Extracts the raw encoded bytes of one glTF image from its buffer view.
│   └── calls _read_buffer_view_bytes
├── def write_glb(gltf: Dict[str, Any], binary_blob: bytes, path: Union[str, Path]) -> None
│   ├── # Serializes a glTF JSON document + binary buffer into the GLB chunked container (12-byte header + JSON chunk + BIN chunk) on disk.
│   ├── impls bin_payload = binary_blob as bytes
│   ├── if gltf declares no buffers
│   │   └── impls gltf["buffers"] = one empty buffer entry
│   ├── impls gltf["buffers"][0]["byteLength"] = bin_payload's length  # the container's own record of the blob it carries
│   ├── impls json_bytes = gltf serialized as compact JSON, utf-8 encoded
│   ├── impls json_chunk = json_bytes space-padded to a four-byte boundary
│   ├── impls bin_chunk = bin_payload zero-padded to a four-byte boundary
│   ├── impls total_length = _GLB_HEADER_SIZE, twice _GLB_CHUNK_HEADER_SIZE, and both chunks' lengths summed  # impls-node-one-step:skip
│   └── with path opened for binary writing as stream
│       ├── impls write _GLB_MAGIC, _GLB_VERSION, and total_length as three little-endian uint32  # impls-node-one-step:skip
│       ├── impls write json_chunk's length with _GLB_JSON_CHUNK_TYPE
│       ├── impls write json_chunk
│       ├── impls write bin_chunk's length with _GLB_BIN_CHUNK_TYPE
│       └── impls write bin_chunk
├── def append_accessor(gltf: Dict[str, Any], binary_blob: bytearray, array: np.ndarray, target: Optional[int]) -> int
│   ├── # Appends an array to the buffer as a new bufferView + accessor (componentType/type inferred from the array dtype/shape), returning the new accessor index.
│   ├── calls _numpy_component_type
│   └── calls _accessor_type
├── def append_image(gltf: Dict[str, Any], binary_blob: bytearray, image_bytes: bytes, mime_type: str) -> int
│   ├── # Appends encoded image bytes to the buffer as a new bufferView + image, returning the new image index.
│   ├── impls offset = binary_blob's current length, padded to a four-byte boundary
│   ├── impls extend binary_blob with image_bytes
│   ├── impls append a bufferView over offset for image_bytes' length
│   ├── impls append an image entry naming that bufferView, carrying mime_type
│   ├── impls image_index = the new image's index into gltf's images
│   └── return image_index
├── def _apply_sparse_overlay(gltf: Dict[str, Any], binary_blob: bytes, sparse: Dict[str, Any], target_array: np.ndarray) -> None
│   ├── # Overwrites the glTF sparse-accessor index/value pairs onto the densely-read accessor values in place.
│   ├── impls sparse_count = sparse["count"]
│   ├── calls _component_dtype         # the sparse indices' componentType
│   ├── calls _read_buffer_view_bytes  # the sparse indices' buffer view
│   ├── impls index_array = sparse_count values from the indices' byteOffset, read as index_dtype, as int64
│   ├── impls value_element_byte_size = target_array's dtype itemsize times its component count
│   ├── calls _read_buffer_view_bytes  # the sparse values' buffer view
│   ├── impls value_array = sparse_count runs of value_element_byte_size from the values' byteOffset, read as target_array's dtype, reshaped to sparse_count by its component count
│   └── impls target_array[index_array] = value_array  # in place, so the caller sees the overlay
├── def _read_buffer_view_bytes(gltf: Dict[str, Any], binary_blob: bytes, buffer_view_index: int) -> bytes
│   ├── # Slices the raw bytes of one glTF buffer view out of the binary blob.
│   ├── impls buffer_view = gltf's bufferViews at buffer_view_index
│   ├── impls offset = buffer_view's byteOffset, defaulting to zero
│   ├── impls view_bytes = binary_blob sliced from offset for buffer_view's byteLength
│   └── return view_bytes
├── def _component_dtype(component_type: int) -> np.dtype
│   ├── # Maps a glTF accessor componentType code to its numpy dtype.
│   ├── impls component_dtype = the numpy dtype the glTF componentType table pairs with component_type
│   └── return component_dtype
├── def _component_count(accessor_type: str) -> int
│   ├── # Maps a glTF accessor type string (SCALAR / VEC2 / VEC3 / ...) to its component count.
│   ├── impls component_count = the component count the glTF accessor-type table pairs with accessor_type
│   └── return component_count
├── def _numpy_component_type(dtype: np.dtype) -> int
│   ├── # Maps a numpy dtype to its glTF accessor componentType code.
│   ├── impls component_type = the componentType code that table pairs with dtype
│   └── return component_type  # the inverse of _component_dtype
└── def _accessor_type(num_components: int) -> str
    ├── # Maps a component count to its glTF accessor type string (SCALAR / VEC2 / VEC3 / ...).
    ├── impls accessor_type = the accessor type string that table pairs with num_components
    └── return accessor_type  # the inverse of _component_count
```

`utils/io/image.py`

```text
image.py
├── from typing import List, Literal, Optional, Sequence, Union
├── import numpy
├── import rasterio
├── import torch
├── import torchvision
├── from PIL import Image
├── from utils.input_checks import check_read_file, check_write_file
├── def load_image(filepath: Optional[str] = None, filepaths: Optional[Union[str, List[str]]] = None, height: Optional[int] = None, width: Optional[int] = None, sub: Optional[Union[float, int, Sequence[float], Sequence[int], torch.Tensor]] = None, div: Optional[Union[float, int, Sequence[float], Sequence[int], torch.Tensor]] = None, normalization: Optional[Literal["min-max", "mean-std"]] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor
│   ├── # Loads one image file, or a set of band files, into a tensor.
│   ├── if filepath and filepaths are both given or both absent
│   │   └── raise ValueError  # exactly one of the two routes is provided
│   ├── if filepath
│   │   └── calls _load_image(filepath)
│   ├── else
│   │   └── calls _load_multispectral_image(filepaths, height, width)
│   ├── calls _normalize(image, sub=sub, div=div, normalization=normalization)
│   ├── if dtype is not None
│   │   ├── if dtype is not a torch.dtype
│   │   │   └── raise TypeError
│   │   └── impls image = image.to(dtype)
│   └── return image
├── def _load_image(filepath: str) -> torch.Tensor
│   ├── # Loads one .png / .jpg / .jpeg / .bmp file into a tensor, dispatching on the PIL mode.
│   ├── calls check_read_file(path=filepath, ext=['.png', '.jpg', '.jpeg', '.bmp'])
│   ├── calls Image.open(filepath)
│   ├── impls mode = the opened image's PIL mode
│   ├── if mode == 'RGB'
│   │   ├── impls image = the PIL array as a tensor
│   │   ├── impls image = it permuted (H, W, C) -> (C, H, W)
│   │   └── impls assert it is a 3-channel uint8 tensor
│   ├── elif mode == 'RGBA'
│   │   ├── impls image = the PIL array as a tensor
│   │   ├── impls image = it permuted (H, W, C) -> (C, H, W)
│   │   └── impls assert it is a 4-channel uint8 tensor
│   ├── elif mode in ['L', 'P']
│   │   ├── impls image = the PIL array as a tensor
│   │   └── impls assert it is a 2-D uint8 tensor
│   ├── elif mode in ['I', 'I;16']
│   │   ├── impls image = the PIL array as an int32 tensor
│   │   └── impls assert it is 2-D
│   ├── else
│   │   └── raise NotImplementedError  # no PIL-to-tensor conversion for this mode
│   └── return image
├── def _load_multispectral_image(filepaths: Union[str, List[str]], height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor
│   ├── # Loads separate .tif band files into one tensor, one channel per band.
│   ├── if filepaths is a str
│   │   └── impls filepaths = [filepaths]
│   ├── impls assert filepaths is a list
│   ├── for each path in filepaths
│   │   └── calls check_read_file(path=path, ext=['.tif', '.tiff'])
│   ├── if filepaths is empty
│   │   └── raise ValueError
│   ├── impls assert height is None or an int
│   ├── impls assert width is None or an int
│   ├── if one filepath is given with a height or a width
│   │   └── raise ValueError  # resizing is defined only for a multi-band load
│   ├── impls bands = an empty list
│   ├── for each path in filepaths
│   │   ├── with rasterio.open(path) as src
│   │   │   └── impls band = src.read()
│   │   ├── if the band is uint16
│   │   │   └── impls band = the band as int64
│   │   ├── impls band_tensor = the band as a tensor
│   │   ├── if band_tensor is not 3-D
│   │   │   └── raise ValueError
│   │   └── impls append band_tensor to bands
│   ├── try
│   │   ├── impls image = torch.cat(bands, dim=0)
│   │   └── return image  # the bands concatenated on the channel dim
│   └── except RuntimeError
│       ├── if height is None
│       │   └── impls height = the bands' maximum height
│       ├── if width is None
│       │   └── impls width = the bands' maximum width
│       ├── impls resized_bands = each band resized to (height, width)
│       ├── impls image = torch.cat(resized_bands, dim=0)
│       └── return image  # the resized bands concatenated on the channel dim
├── def _normalize(image: torch.Tensor, sub: Optional[Union[float, int, Sequence[float], Sequence[int], torch.Tensor]], div: Optional[Union[float, int, Sequence[float], Sequence[int], torch.Tensor]], normalization: Optional[str] = "min-max") -> torch.Tensor
│   ├── # Normalizes the image by subtraction / division, or by min-max or mean-std statistics.
│   ├── impls assert image is a torch.Tensor
│   ├── impls assert normalization is None, "min-max" or "mean-std"
│   ├── impls assert sub and div are each None, a float, an int, a list, a tuple or a tensor  # impls-node-one-step:skip
│   ├── if normalization is set together with sub or div
│   │   └── raise ValueError  # the two normalization routes are mutually exclusive
│   ├── if sub, div and normalization are all None
│   │   └── return image  # unchanged
│   ├── impls image = image.to(torch.float32)
│   ├── def prepare_tensor(value, num_channels, ndim) [local]
│   │   ├── # Prepares a broadcastable normalization tensor from a scalar or per-channel value.
│   │   ├── impls value_tensor = value as a float32 tensor
│   │   ├── if its element count is neither one nor num_channels
│   │   │   └── raise ValueError
│   │   ├── if its element count is one
│   │   │   └── impls value_tensor = it expanded to num_channels
│   │   ├── impls broadcast_tensor = value_tensor viewed as (-1, 1, 1) for a 3-D image, else unchanged
│   │   └── return broadcast_tensor
│   ├── if sub is not None
│   │   ├── calls prepare_tensor(sub, the image's channel count when 3-D else 1, image.ndim)
│   │   └── impls image = image - sub_tensor
│   ├── if div is not None
│   │   ├── calls prepare_tensor(div, the image's channel count when 3-D else 1, image.ndim)
│   │   ├── impls div_tensor = it, with magnitudes below 1e-6 raised to 1e-6  # avoids dividing by zero
│   │   └── impls image = image / div_tensor
│   ├── if normalization == "min-max"
│   │   └── impls image = (image - its min) / (its max - its min + 1e-6)
│   ├── elif normalization == "mean-std"
│   │   ├── impls std_val = the image's std, raised to 1e-6 when below it  # avoids dividing by zero
│   │   └── impls image = (image - the image's mean) / std_val
│   └── return image
├── def save_image(tensor: torch.Tensor, filepath: str) -> None
│   ├── # Writes a tensor to an image file, dispatching on its shape and dtype.
│   ├── calls check_write_file(path=filepath)
│   ├── if tensor is a 3-channel float32 CHW tensor
│   │   └── calls torchvision.utils.save_image(tensor=tensor, fp=filepath)
│   ├── elif tensor is a 2-D uint8 tensor
│   │   ├── calls Image.fromarray(tensor.numpy())
│   │   └── impls save that PIL image to filepath
│   └── else
│       └── raise NotImplementedError  # the shape / dtype pair has no writer
├── def decode_image_bytes(image_bytes: bytes) -> torch.Tensor
│   ├── # Decodes encoded image bytes (PNG / JPEG / ...) into an HWC uint8 RGB tensor — in-memory counterpart of the file-based load_image.
│   ├── impls array = image_bytes decoded by the imaging library into an HWC uint8 RGB array
│   ├── impls image = array as a tensor
│   └── return image  # (H, W, 3) uint8, RGB
└── def encode_image_bytes(image: torch.Tensor, image_format: str) -> bytes
    ├── # Encodes an HWC image tensor into encoded image bytes (PNG / JPEG / ...) — in-memory counterpart of the file-based save_image.
    ├── impls buffer = an in-memory bytes buffer
    ├── impls save image, as a PIL RGB image, into buffer under image_format
    ├── impls encoded_image = buffer's bytes
    └── return encoded_image  # the encoded image
```

`utils/io/chumpy.py`

```text
chumpy.py
├── import pickle
├── import numpy as np
├── _CHUMPY_ROOT_MODULE  # str / "chumpy", the package whose classes the stream names and this project does not install
├── def load_chumpy(filepath: str) -> Dict[str, Any]
│   ├── # Loads a chumpy-pickled file as plain numpy arrays, chumpy being an abandoned Python-2-era package that neither builds nor imports against a modern numpy.
│   ├── def _validate_inputs [local]
│   │   └── impls assert the filepath's extension is .pkl  # the one form this loader reads, which is what leaves the dispatch below no unhandled case
│   ├── calls _validate_inputs
│   ├── if the filepath's extension is .pkl
│   │   ├── calls _load_chumpy_from_pkl
│   │   └── return  # the mapping, each of the file's chumpy arrays now a numpy one
│   └── assert 0, "Should not reach here."
├── def _load_chumpy_from_pkl(filepath: str) -> Dict[str, Any]
│   ├── # Reads the pickle with every chumpy class the stream names resolved to _ChumpyArray, then resolves those stand-ins to the arrays they carry.
│   ├── with filepath opened for binary reading
│   │   ├── calls _ChumpyUnpickler(file=the opened file, encoding="latin1")  # the file is a Python-2 pickle
│   │   └── calls unpickler.load
│   ├── for each name and value in the unpickled mapping
│   │   └── impls replace a _ChumpyArray value by the array it carries, leaving every other value as the stream produced it  # a scipy sparse entry stays sparse
│   └── return  # the mapping, carrying no stand-in beyond this function
├── class _ChumpyUnpickler(pickle.Unpickler)
│   ├── # Unpickler resolving chumpy's own classes to _ChumpyArray, confining the substitution to this load rather than to global state.
│   └── def find_class(self, module: str, name: str) -> Any   [override]
│       ├── # Resolves one class reference the stream names, substituting the stand-in for chumpy's own.
│       ├── if module is _CHUMPY_ROOT_MODULE or one of its submodules
│       │   └── return _ChumpyArray
│       └── return  # whatever the base unpickler resolves the reference to
└── class _ChumpyArray
    ├── # Stand-in for one chumpy array while unpickling: it takes the pickled state and yields the numpy array already inside it.
    ├── def __setstate__(self, state: Any) -> None
    │   ├── # Restores the pickled chumpy state onto the stand-in.
    │   └── impls update self.__dict__ from state, or from {"_state": state} when the payload is not a dict
    └── def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray
        ├── # Yields the stored numpy array, the hook that makes the stand-in coercible.
        ├── for each value in self.__dict__.values()
        │   └── if the value is an np.ndarray
        │       └── return  # the value, cast to dtype when one is given
        └── raise TypeError  # the state carries no array
```
