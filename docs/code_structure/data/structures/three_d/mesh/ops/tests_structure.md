# data/structures/three_d/mesh/ops — tests structure

`tests/data/structures/three_d/mesh/ops/test_apply_transform.py`

```text
test_apply_transform.py
├── def test_verts_match_reference_matmul
│   └── # transformed verts equal a direct homogeneous matmul of mesh.verts by the transform.
├── def test_faces_and_texture_preserved
│   └── # the returned Mesh keeps the original faces and texture unchanged.
└── def test_rejects_non_4x4_transform
    └── # a transform that is not a [4, 4] matrix raises an assertion.
```
