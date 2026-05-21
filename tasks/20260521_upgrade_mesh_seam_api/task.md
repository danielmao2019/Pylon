goal: upgrade Mesh class API to deal with seamed UV atlas

## design

1. **Texture is extracted from `Mesh` into a dedicated `MeshTexture` type.** `Mesh`
   holds `texture: Optional[MeshTexture]`; `mesh.texture is None` means
   geometry-only. `MeshTexture` is an abstract base with two concrete
   representations — `MeshTextureVertexColor` (per-vertex RGB) and
   `MeshTextureUVTextureMap` (UV atlas image + `vertex_uv` + `face_uvs` +
   UV-origin convention). The active representation is the texture object's
   concrete type, checked by `isinstance`, not a discriminant field.

2. **Canonical, seam-invariant vertex domain (the seam contract).** `Mesh.vertices`
   / `Mesh.faces` always hold the **geometry domain** — the distinct surface
   positions and the faces indexing them — and that domain is independent of
   which loader produced the mesh, so `len(mesh.vertices)` is loader-independent
   for a given asset. A UV-atlas seam never inflates this domain: the seam is
   confined entirely to the UV layer of `MeshTextureUVTextureMap` (`vertex_uv` /
   `face_uvs`), where `U >= V`. Any loader or converter that receives a mesh in
   the **per-corner-expanded** form — one vertex per `v/vt` pair, `V == U`, as
   `trimesh.load(force="mesh")` produces by duplicating seam vertices — MUST
   **weld** those coincident per-corner duplicates back to the geometry domain
   on ingest, re-expressing the seam as `vertex_uv` / `face_uvs`. Welding is by
   exact-position equality (trimesh's seam duplicates are exact copies, so the
   weld is lossless). PyTorch3D `load_obj` already yields the decoupled
   geometry/UV domains, so the OBJ loader needs no welding; `mesh_from_trimesh`
   is the entry path that must weld.

## intentional changes

1. The shared `_assert_rgb_range` validation helper (present in the pre-refactor `data/structures/three_d/mesh/validate.py`) is intentionally removed. The refactor splits attribute validation into per-representation modules under `texture/` (`validate_vertex_color.py`, `validate_uv_texture_map.py`); each module fully owns its representation's validation, including the float32 RGB `[0, 1]` range assertion for its own data. Once validators are split by representation a shared range helper is not a meaningful abstraction, and it has no clean home — keeping it would force a `texture/` -> `mesh/validate.py` import dependency, and the `MeshTexture` ABC module is not a validation home. This is a deliberate part of the refactor, not an accidental omission.
