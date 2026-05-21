goal: upgrade Mesh class API to deal with seamed UV atlas

## intentional changes

1. The shared `_assert_rgb_range` validation helper (present in the pre-refactor `data/structures/three_d/mesh/validate.py`) is intentionally removed. The refactor splits attribute validation into per-representation modules under `texture/` (`validate_vertex_color.py`, `validate_uv_texture_map.py`); each module fully owns its representation's validation, including the float32 RGB `[0, 1]` range assertion for its own data. Once validators are split by representation a shared range helper is not a meaningful abstraction, and it has no clean home — keeping it would force a `texture/` -> `mesh/validate.py` import dependency, and the `MeshTexture` ABC module is not a validation home. This is a deliberate part of the refactor, not an accidental omission.
