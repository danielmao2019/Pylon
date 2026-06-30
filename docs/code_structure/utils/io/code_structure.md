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
