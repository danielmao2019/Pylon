# Utils IO Folder Structure

## 1. Folder structure trees

`./utils/io/`

```text
io/
├── __init__.py            # io package API surface
├── config.py              # config-file I/O
├── glb.py                 # generic glTF/GLB I/O — the chunked file, the typed accessor arrays it encodes, and the embedded raw image bytes
├── image.py               # image read / write (files) + in-memory bytes codec (decode_image_bytes / encode_image_bytes)
├── json.py                # JSON read / write
├── octree_gs_scaffold.py  # octree Gaussian-splatting scaffold I/O
├── torch.py               # torch tensor (de)serialization I/O
└── xmp.py                 # XMP metadata I/O
```

Only `glb.py` is detailed in `code_structure.md`; the sibling modules predate this skeleton and are listed for placement only.
