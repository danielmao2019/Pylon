# Utils IO Code Structure

## glTF/GLB I/O

`utils/io/glb.py`

```text
glb.py
├── import json
├── import struct
├── import numpy as np
├── def load_glb_json_and_bin(path: Union[str, Path]) -> Tuple[Dict[str, Any], bytes]
│   └── # Parses a GLB container into its glTF JSON document and binary buffer blob.
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
│   └── # Serializes a glTF JSON document + binary buffer into the GLB chunked container (12-byte header + JSON chunk + BIN chunk) on disk.
├── def append_accessor(gltf: Dict[str, Any], binary_blob: bytearray, array: np.ndarray, target: Optional[int]) -> int
│   ├── # Appends an array to the buffer as a new bufferView + accessor (componentType/type inferred from the array dtype/shape), returning the new accessor index.
│   ├── calls _numpy_component_type
│   └── calls _accessor_type
├── def append_image(gltf: Dict[str, Any], binary_blob: bytearray, image_bytes: bytes, mime_type: str) -> int
│   └── # Appends encoded image bytes to the buffer as a new bufferView + image, returning the new image index.
├── def _apply_sparse_overlay(accessor: Dict[str, Any], gltf: Dict[str, Any], binary_blob: bytes, dense_values: np.ndarray) -> None
│   └── # Overwrites the glTF sparse-accessor index/value pairs onto the densely-read accessor values in place.
├── def _read_buffer_view_bytes(gltf: Dict[str, Any], binary_blob: bytes, buffer_view_index: int) -> bytes
│   └── # Slices the raw bytes of one glTF buffer view out of the binary blob.
├── def _component_dtype(component_type: int) -> np.dtype
│   └── # Maps a glTF accessor componentType code to its numpy dtype.
├── def _component_count(accessor_type: str) -> int
│   └── # Maps a glTF accessor type string (SCALAR / VEC2 / VEC3 / ...) to its component count.
├── def _numpy_component_type(dtype: np.dtype) -> int
│   └── # Maps a numpy dtype to its glTF accessor componentType code.
└── def _accessor_type(num_components: int) -> str
    └── # Maps a component count to its glTF accessor type string (SCALAR / VEC2 / VEC3 / ...).
```

## Image: in-memory bytes codec

`utils/io/image.py`

```text
image.py
├── def decode_image_bytes(image_bytes: bytes) -> torch.Tensor
│   └── # Decodes encoded image bytes (PNG / JPEG / ...) into an HWC uint8 RGB tensor — in-memory counterpart of the file-based load_image.
└── def encode_image_bytes(image: torch.Tensor, image_format: str) -> bytes
    └── # Encodes an HWC image tensor into encoded image bytes (PNG / JPEG / ...) — in-memory counterpart of the file-based save_image.
```

## chumpy pickle I/O

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
