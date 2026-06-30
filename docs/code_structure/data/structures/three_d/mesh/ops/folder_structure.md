# data/structures/three_d/mesh/ops — folder structure

## Code folder structure

```text
data/structures/three_d/mesh/ops/
├── __init__.py
├── apply_transform.py             # maps a Mesh's verts through a 4x4 transform via the chunked large×small matmul
└── world_to_camera_transform.py   # high-level world-to-camera mesh transform, implemented via apply_transform
```

## Tests folder structure

```text
tests/data/structures/three_d/mesh/ops/
└── test_apply_transform.py    # transformed verts match a reference homogeneous matmul; faces and texture preserved
```
