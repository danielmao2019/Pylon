# Utils IO Folder Structure

## Code folder structure

```text
./utils/io/
├── __init__.py            # io package API surface
├── config.py              # config-file I/O
├── glb.py                 # generic glTF/GLB I/O — the chunked file, the typed accessor arrays it encodes, and the embedded raw image bytes
├── image.py               # image read / write (files) + in-memory bytes codec (decode_image_bytes / encode_image_bytes)
├── json.py                # JSON read / write
├── octree_gs_scaffold.py  # octree Gaussian-splatting scaffold I/O
├── torch.py               # torch tensor (de)serialization I/O
└── xmp.py                 # XMP metadata I/O
```

## Tests folder structure

```text
./tests/utils/io/
├── config/
│   └── test_config_loading.py
├── image/
│   ├── test_image_loading.py
│   └── test_image_saving.py
├── json/
│   ├── test_json_loading.py
│   └── test_json_saving.py
├── point_clouds/
│   ├── load_point_cloud/
│   │   ├── test_point_cloud_loading.py
│   │   ├── test_point_cloud_operations.py
│   │   └── test_precision_handling.py
│   └── save_point_cloud/
│       └── test_ply_saving.py
└── torch/
    ├── test_torch_loading.py
    └── test_torch_saving.py
```
