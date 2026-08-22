# Mesh Data Structure Tests Structure

## 1. Code structure trees

`tests/data/structures/three_d/mesh/test_convert.py`

```text
test_convert.py
├── import sys
├── import types
├── from pathlib import Path
├── import numpy as np
├── import open3d as o3d
├── import torch
├── import trimesh
├── from PIL import Image
├── REPO_ROOT  # Path: the repo root, five parents up from this test file
├── calls _install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
├── calls _install_namespace_package(package_name="data.structures", package_path=REPO_ROOT / "data" / "structures")
├── calls _install_namespace_package(package_name="data.structures.three_d", package_path=REPO_ROOT / "data" / "structures" / "three_d")
├── from data.structures.three_d.mesh import Mesh, MeshTextureUVTextureMap, MeshTextureVertexColor, mesh_from_open3d, mesh_from_pytorch3d, mesh_from_trimesh, mesh_to_open3d, mesh_to_pytorch3d, mesh_to_trimesh
├── def test_mesh_from_trimesh_welds_seam_to_geometry_domain(tmp_path: Path) -> None
│   ├── # A seamed UV mesh that trimesh loads in per-corner-expanded form (V == U) must come through mesh_from_trimesh on the canonical geometry domain (V <= U, distinct positions), with the seam carried only by verts_uvs / faces_uvs.
│   ├── calls _write_seamed_uv_obj                                 # into tmp_path -> the seamed OBJ path
│   ├── calls trimesh.load(that OBJ, force="mesh", process=False)  # -> the source mesh
│   ├── assert the source mesh carries a UV visual
│   ├── assert the source mesh has 6 vertices                             # trimesh's per-corner-expanded form
│   ├── calls mesh_from_trimesh(mesh=that source mesh, convention="obj")  # -> the welded mesh
│   ├── assert the result is a Mesh
│   ├── assert its texture is a MeshTextureUVTextureMap
│   ├── impls vertex_count — the welded mesh's verts row count
│   ├── impls uv_count — the welded texture's verts_uvs row count
│   ├── assert vertex_count == 4
│   ├── assert uv_count == 6
│   ├── assert vertex_count <= uv_count                   # the canonical geometry domain
│   ├── calls torch.unique(over the welded verts, dim=0)  # -> the distinct positions
│   ├── assert the distinct-position count equals vertex_count
│   └── assert the welded faces_uvs shape equals the welded faces shape
├── def test_vertex_count_is_loader_independent(tmp_path: Path) -> None
│   ├── # For one OBJ asset, len(mesh.verts) must be identical whether the mesh is loaded via Mesh.load (PyTorch3D) or via mesh_from_trimesh, since both land on the canonical geometry domain.
│   ├── calls _write_seamed_uv_obj  # into tmp_path -> the seamed OBJ path
│   ├── calls Mesh.load             # that path -> the PyTorch3D-loaded mesh
│   ├── calls trimesh.load(the same OBJ, force="mesh", process=False)
│   ├── calls mesh_from_trimesh(mesh=that trimesh mesh, convention="obj")  # -> the trimesh-loaded mesh
│   ├── assert both loaded meshes report the same verts row count
│   └── assert the PyTorch3D-loaded mesh has 4 verts
├── def test_trimesh_uv_round_trip_preserves_geometry() -> None
│   ├── # mesh_to_trimesh then mesh_from_trimesh must preserve geometry, UV, and texture (expand then weld is identity on the geometry domain).
│   ├── calls _build_uv_textured_mesh  # -> the source mesh
│   ├── calls mesh_to_trimesh          # that mesh -> the trimesh mesh
│   ├── calls mesh_from_trimesh(mesh=that trimesh mesh, convention="obj")  # -> the round-tripped mesh
│   ├── assert the round-tripped texture is a MeshTextureUVTextureMap
│   ├── assert the round-tripped verts row count equals the source's
│   ├── impls sorted_original — the source verts reordered by argsort over x * 1.0e06 + y
│   ├── impls sorted_round_trip — the round-tripped verts reordered by the same key
│   ├── calls torch.testing.assert_close  # sorted_round_trip against sorted_original — the vert ORDER is not preserved, and faces are not compared at all
│   ├── calls torch.testing.assert_close  # the round-tripped uv_texture_map against the source's
│   ├── impls original_uv_by_face — the source verts_uvs gathered at faces_uvs.reshape(-1), one UV per face corner
│   ├── impls round_trip_uv_by_face — the round-tripped verts_uvs gathered the same way
│   └── calls torch.testing.assert_close  # round_trip_uv_by_face against original_uv_by_face — UV is compared per face corner, not per verts_uvs row
├── def test_pytorch3d_round_trip_preserves_texture() -> None
│   ├── # mesh_to_pytorch3d then mesh_from_pytorch3d must preserve geometry and texture for both vertex-colored and UV-textured meshes.
│   ├── calls _build_vertex_color_mesh  # -> the vertex-colored source mesh
│   ├── calls mesh_to_pytorch3d         # that mesh, device cpu -> the PyTorch3D mesh
│   ├── calls mesh_from_pytorch3d(mesh=the vertex-colored PyTorch3D mesh, convention="obj")  # -> the round-tripped mesh
│   ├── assert its texture is a MeshTextureVertexColor
│   ├── calls torch.testing.assert_close  # the round-tripped verts against the source's
│   ├── assert the round-tripped faces equal the source's, elementwise
│   ├── calls torch.testing.assert_close  # the round-tripped vertex_color against the source's
│   ├── calls _build_uv_textured_mesh     # -> the UV-textured source mesh
│   ├── calls mesh_to_pytorch3d           # that mesh, device cpu -> the PyTorch3D mesh
│   ├── calls mesh_from_pytorch3d(mesh=the UV-textured PyTorch3D mesh, convention="obj")  # -> the round-tripped mesh
│   ├── assert its texture is a MeshTextureUVTextureMap
│   ├── calls torch.testing.assert_close  # the round-tripped verts against the source's
│   ├── assert the round-tripped faces equal the source's, elementwise
│   ├── calls torch.testing.assert_close  # the round-tripped uv_texture_map against the source's
│   ├── calls torch.testing.assert_close  # the round-tripped verts_uvs against the source's
│   └── assert the round-tripped faces_uvs equal the source's, elementwise
├── def test_open3d_round_trip_preserves_vertex_color() -> None
│   ├── # mesh_to_open3d then mesh_from_open3d must preserve geometry and vertex colors (the Open3D path carries no UV texture).
│   ├── calls _build_vertex_color_mesh  # -> the source mesh
│   ├── calls mesh_to_open3d            # that mesh -> the Open3D mesh
│   ├── calls mesh_from_open3d          # that Open3D mesh -> the round-tripped mesh
│   ├── assert its texture is a MeshTextureVertexColor
│   ├── calls torch.testing.assert_close  # the round-tripped verts against the source's
│   ├── assert the round-tripped faces equal the source's, elementwise
│   └── calls torch.testing.assert_close  # the round-tripped vertex_color against the source's
├── def _install_namespace_package(package_name: str, package_path: Path) -> None
│   ├── # Installs one namespace package into sys.modules so the data tree imports without repo-level setup.
│   ├── if package_name is already in sys.modules
│   │   └── return  # already installed, nothing to do
│   ├── impls module — a fresh types.ModuleType named package_name
│   ├── impls point its __file__ at package_path / "__init__.py"
│   ├── impls point its __path__ at package_path
│   └── impls register it in sys.modules under package_name
├── def _write_seamed_uv_obj(directory: Path) -> Path
│   ├── # Writes one seamed unit-square OBJ — 4 distinct positions, 6 UVs, 2 triangles — with its sibling MTL and texture PNG, so the suite has a real UV seam to load.
│   ├── impls obj_path, mtl_path and texture_path under directory
│   ├── calls obj_path.write_text(the seam-crossing OBJ text, encoding="utf-8")  # 4 v, 6 vt, 2 f rows over material0
│   ├── calls mtl_path.write_text                      # newmtl material0 with map_Kd seam_texture.png
│   ├── calls np.full((4, 4, 3), 128, dtype=np.uint8)  # -> the flat mid-grey texture pixels
│   ├── calls Image.fromarray(that array)              # -> the PIL image
│   ├── impls save that image to texture_path
│   └── return  # obj_path
├── def _build_vertex_color_mesh() -> Mesh
│   ├── # Offers one CPU-owned vertex-colored single-triangle mesh, the source the PyTorch3D and Open3D round-trips start from.
│   ├── calls MeshTextureVertexColor(vertex_color=torch.tensor( [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, ))
│   ├── calls Mesh(verts=a float32 [3, 3], faces=an int64 [1, 3], texture=a MeshTextureVertexColor over a float32 [3, 3] vertex_color)
│   └── return  # that mesh
└── def _build_uv_textured_mesh() -> Mesh
    ├── # Offers one CPU-owned UV-textured single-triangle mesh already on the geometry domain, the source the trimesh and PyTorch3D UV round-trips start from.
    ├── calls MeshTextureUVTextureMap  # a 2x2x3 float32 map, verts_uvs {(0.1, 0.1), (0.4, 0.1), (0.1, 0.4)}, faces_uvs [[0, 1, 2]], convention "obj"
    ├── calls Mesh(verts=a float32 [3, 3], faces=an int64 [1, 3], texture=a MeshTextureUVTextureMap over a float32 [2, 2, 3] map, a float32 [3, 2] verts_uvs, an int64 [1, 3] faces_uvs, convention "obj")
    └── return  # that mesh
```

`tests/data/structures/three_d/mesh/texture/test_conventions.py`

```text
test_conventions.py
├── import sys
├── import types
├── from pathlib import Path
├── import torch
├── REPO_ROOT  # Path: the repo root, six parents up from this test file
├── calls _install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
├── calls _install_namespace_package(package_name="data.structures", package_path=REPO_ROOT / "data" / "structures")
├── calls _install_namespace_package(package_name="data.structures.three_d", package_path=REPO_ROOT / "data" / "structures" / "three_d")
├── from data.structures.three_d.mesh.texture.conventions import transform_convention
├── def test_identity_when_conventions_match() -> None
│   ├── # transform_convention returns the UV table unchanged when the source and target conventions are equal.
│   ├── impls verts_uvs — a float32 [3, 2] table of {(0.0, 0.0), (1.0, 0.25), (0.5, 1.0)}
│   ├── calls transform_convention(verts_uvs=that table, source_convention="obj", target_convention="obj")  # -> the transformed table
│   └── assert the transformed table IS the input object  # identity, not merely elementwise equality
├── def test_flips_v_axis_when_conventions_differ() -> None
│   ├── # transform_convention flips the V axis (v -> 1 - v) when the source and target conventions differ.
│   ├── impls verts_uvs — a float32 [3, 2] table of {(0.0, 0.0), (1.0, 0.25), (0.5, 1.0)}
│   ├── calls transform_convention(verts_uvs=that table, source_convention="obj", target_convention="top_left")  # -> the transformed table
│   ├── assert the transformed table matches {(0.0, 1.0), (1.0, 0.75), (0.5, 0.0)} within atol 1.0e-06
│   └── assert the transformed u column equals the source u column, elementwise
└── def _install_namespace_package(package_name: str, package_path: Path) -> None
    ├── # Installs one namespace package into sys.modules so the data tree imports without repo-level setup.
    ├── if package_name is already in sys.modules
    │   └── return  # already installed, nothing to do
    ├── impls module — a fresh types.ModuleType named package_name
    ├── impls point its __file__ at package_path / "__init__.py"
    ├── impls point its __path__ at package_path
    └── impls register it in sys.modules under package_name
```

`tests/data/structures/three_d/mesh/texture/test_mesh_texture_vertex_color.py`

```text
test_mesh_texture_vertex_color.py
├── import sys
├── import types
├── from pathlib import Path
├── import pytest
├── import torch
├── REPO_ROOT  # Path: the repo root, six parents up from this test file
├── calls _install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
├── calls _install_namespace_package(package_name="data.structures", package_path=REPO_ROOT / "data" / "structures")
├── calls _install_namespace_package(package_name="data.structures.three_d", package_path=REPO_ROOT / "data" / "structures" / "three_d")
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def test_normalizes_uint8_to_float01() -> None
│   ├── # MeshTextureVertexColor normalizes a uint8 [0,255] vertex_color into contiguous float32 [V,3] in [0,1].
│   ├── calls MeshTextureVertexColor  # a uint8 [3, 3] vertex_color of {(255, 0, 0), (0, 128, 0), (0, 0, 255)}
│   ├── assert the stored vertex_color dtype is float32
│   ├── assert its shape is (3, 3)
│   ├── assert it is contiguous
│   └── assert it matches {(1.0, 0.0, 0.0), (0.0, 128/255, 0.0), (0.0, 0.0, 1.0)} within atol 1.0e-06
├── def test_rejects_out_of_range_float() -> None
│   ├── # MeshTextureVertexColor rejects a float32 vertex_color carrying values outside [0,1].
│   └── with pytest.raises(AssertionError)  # matching "at most 1"
│       └── calls MeshTextureVertexColor(vertex_color=torch.tensor( [[1.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, ))
├── def test_to_rejects_non_none_convention() -> None
│   ├── # MeshTextureVertexColor.to raises when given a non-None convention, since vertex color carries no UV convention.
│   ├── calls MeshTextureVertexColor(vertex_color=torch.tensor( [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, ))
│   └── with pytest.raises(AssertionError)  # matching "convention"
│       └── calls texture.to(convention="obj")
└── def _install_namespace_package(package_name: str, package_path: Path) -> None
    ├── # Installs one namespace package into sys.modules so the data tree imports without repo-level setup.
    ├── if package_name is already in sys.modules
    │   └── return  # already installed, nothing to do
    ├── impls module — a fresh types.ModuleType named package_name
    ├── impls point its __file__ at package_path / "__init__.py"
    ├── impls point its __path__ at package_path
    └── impls register it in sys.modules under package_name
```

`tests/data/structures/three_d/mesh/texture/test_mesh_texture_uv_texture_map.py`

```text
test_mesh_texture_uv_texture_map.py
├── import sys
├── import types
├── from pathlib import Path
├── import pytest
├── import torch
├── REPO_ROOT  # Path: the repo root, six parents up from this test file
├── calls _install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
├── calls _install_namespace_package(package_name="data.structures", package_path=REPO_ROOT / "data" / "structures")
├── calls _install_namespace_package(package_name="data.structures.three_d", package_path=REPO_ROOT / "data" / "structures" / "three_d")
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── def test_rejects_faces_uvs_index_out_of_range() -> None
│   ├── # MeshTextureUVTextureMap rejects faces_uvs whose indices do not reference valid verts_uvs rows (the cross-field invariant).
│   └── with pytest.raises(AssertionError)  # matching "verts_uvs"
│       ├── calls _build_uv_texture_map
│       ├── calls _build_seam_safe_verts_uvs
│       └── calls MeshTextureUVTextureMap(uv_texture_map=_build_uv_texture_map(), verts_uvs=_build_seam_safe_verts_uvs(), faces_uvs=torch.tensor([[0, 1, 3]], dtype=torch.int64), convention="obj")
├── def test_normalizes_uint8_texture_map() -> None
│   ├── # MeshTextureUVTextureMap normalizes a uint8 uv_texture_map into contiguous float32 HWC in [0,1].
│   ├── calls _build_seam_safe_verts_uvs
│   ├── calls MeshTextureUVTextureMap(uv_texture_map=a uint8 [2, 2, 3], verts_uvs=_build_seam_safe_verts_uvs(), faces_uvs=an int64 [1, 3], convention="obj")
│   ├── assert the stored uv_texture_map dtype is float32
│   ├── assert its shape is (2, 2, 3)
│   ├── assert it is contiguous
│   ├── assert its max is at most 1.0
│   └── assert texel [0, 0] matches (1.0, 0.0, 0.0) within atol 1.0e-06
├── def test_accepts_seam_safe_verts_uvs_outside_unit_interval() -> None
│   ├── # MeshTextureUVTextureMap accepts verts_uvs whose u extends beyond 1.0 when each face is non-wrapping (its largest cyclic gap is the wraparound gap), the seam-safe canonical form.
│   ├── calls _build_uv_texture_map
│   ├── calls MeshTextureUVTextureMap                # verts_uvs {(0.95, 0.20), (1.05, 0.25), (1.02, 0.80)}, faces_uvs [[0, 1, 2]], convention "obj"
│   └── assert the stored verts_uvs max exceeds 1.0  # the only assertion: the beyond-1.0 u survived construction
├── def test_accepts_wide_non_wrapping_face() -> None
│   ├── # MeshTextureUVTextureMap accepts a wide face whose u-span exceeds 0.5 but whose corners are contiguous (largest cyclic gap is the wraparound gap), e.g. corner u's {0.293, 0.735, 0.801} — a wide face is not a wrapping face.
│   ├── calls _build_uv_texture_map
│   ├── calls MeshTextureUVTextureMap  # verts_uvs {(0.293, 0.20), (0.735, 0.25), (0.801, 0.80)}, faces_uvs [[0, 1, 2]], convention "obj"
│   ├── impls face_u — the stored verts_uvs u column gathered at the int64 faces_uvs
│   ├── impls span — the largest per-face u max-minus-min
│   └── assert span > 0.5  # "test fixture must be a wide face", reporting span
├── def test_rejects_wrapping_face() -> None
│   ├── # MeshTextureUVTextureMap rejects a face whose largest cyclic gap is an interior gap (its corners straddle the cylindrical wrap and were not seam-shifted into contiguous canonical form).
│   └── with pytest.raises(AssertionError)  # matching "non-wrapping"
│       ├── calls _build_uv_texture_map
│       └── calls MeshTextureUVTextureMap  # verts_uvs {(0.95, 0.20), (0.05, 0.25), (0.02, 0.80)}, faces_uvs [[0, 1, 2]], convention "obj"
├── def test_to_converts_uv_convention() -> None
│   ├── # MeshTextureUVTextureMap.to(convention=...) returns a texture whose verts_uvs is converted to the target UV-origin convention.
│   ├── calls _build_uv_texture_map
│   ├── calls _build_seam_safe_verts_uvs
│   ├── calls MeshTextureUVTextureMap(uv_texture_map=_build_uv_texture_map(), verts_uvs=_build_seam_safe_verts_uvs(), faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64), convention="obj")
│   ├── calls texture.to(convention="top_left")  # -> the converted texture
│   ├── assert the converted texture's convention is "top_left"
│   ├── assert its verts_uvs matches {(0.1, 0.9), (0.4, 0.9), (0.1, 0.6)} within atol 1.0e-06
│   ├── assert its faces_uvs equal the source's, elementwise
│   └── assert its uv_texture_map equals the source's, elementwise
├── def _install_namespace_package(package_name: str, package_path: Path) -> None
│   ├── # Installs one namespace package into sys.modules so the data tree imports without repo-level setup.
│   ├── if package_name is already in sys.modules
│   │   └── return  # already installed, nothing to do
│   ├── impls module — a fresh types.ModuleType named package_name
│   ├── impls point its __file__ at package_path / "__init__.py"
│   ├── impls point its __path__ at package_path
│   └── impls register it in sys.modules under package_name
├── def _build_uv_texture_map() -> torch.Tensor
│   ├── # Offers the one small float32 [2, 2, 3] uv_texture_map every constructor case in this file reuses, so no test hand-rolls a map.
│   ├── impls one float32 [2, 2, 3] tensor of four RGB texels
│   └── return  # that map
└── def _build_seam_safe_verts_uvs() -> torch.Tensor
    ├── # Offers one seam-safe verts_uvs table whose only face has u-span <= 0.5, the uncontroversial baseline the seam cases are contrasted against.
    ├── impls one float32 [3, 2] tensor of {(0.1, 0.1), (0.4, 0.1), (0.1, 0.4)}
    └── return  # that UV table
```

`tests/data/structures/three_d/mesh/texture/test_texel_face_map.py`

```text
test_texel_face_map.py
├── import sys
├── import types
├── from pathlib import Path
├── import pytest
├── import torch
├── REPO_ROOT  # Path: the repo root, six parents up from this test file
├── calls _install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
├── calls _install_namespace_package(package_name="data.structures", package_path=REPO_ROOT / "data" / "structures")
├── calls _install_namespace_package(package_name="data.structures.three_d", package_path=REPO_ROOT / "data" / "structures" / "three_d")
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.texel_face_map import build_texel_face_map
├── pytestmark    # pytest.mark.skipif: skips this whole module off CUDA, since build_texel_face_map uses nvdiffrast's CUDA rasterizer
├── _CUDA_DEVICE  # torch.device("cuda"): the one device every mesh in this module is built on
├── def test_build_texel_face_map_returns_texel_face_index_and_barycentric() -> None
│   ├── # build_texel_face_map returns texel_face_index [T, T] int64 and texel_face_barycentric [T, T, 3] float32 with the expected shapes and -1 / NaN sentinels at unoccupied texels.
│   ├── calls _build_identity_uv_mesh                               # -> the mesh under test
│   ├── calls build_texel_face_map(mesh=that mesh, texture_size=8)  # -> the texel-face map
│   ├── assert the map's keys are exactly texel_face_index and texel_face_barycentric
│   ├── impls texel_face_index off that map
│   ├── impls texel_face_barycentric off that map
│   ├── assert texel_face_index has shape (8, 8)
│   ├── assert its dtype is int64
│   ├── assert texel_face_barycentric has shape (8, 8, 3)
│   ├── assert its dtype is float32
│   └── assert at least one texel_face_index entry is -1  # "Expected at least one unoccupied texel", reporting texel_face_index — the -1 sentinel is checked, the NaN one is not
├── def test_build_texel_face_map_maps_identity_face_to_top_row() -> None
│   ├── # On one identity-UV face with small-v corners, the returned texel_face_index assigns face 0 to the top texel rows (top_left v-convention is the rasterizer-buffer mapping).
│   ├── calls _build_identity_uv_mesh                               # -> the mesh under test
│   ├── calls build_texel_face_map(mesh=that mesh, texture_size=8)  # -> the texel-face map
│   ├── impls texel_face_index off that map
│   └── assert at least one entry of the top texel row is face 0  # existential over row 0, reporting that row — not that every face-0 texel lies in the top rows
├── def test_build_texel_face_map_covers_both_sides_of_cylindrical_seam() -> None
│   ├── # For a seam-safe canonical mesh whose only face spans u in {0.95, 1.05, 1.02}, both the u-near-1 and u-near-0 texel columns get assigned to that face (cylindrical wrap coverage via internal seam-side duplication).
│   ├── calls MeshTextureUVTextureMap  # a 1x1x3 zero map, verts_uvs {(0.95, 0.20), (1.05, 0.25), (1.02, 0.80)}, faces_uvs [[0, 1, 2]], convention "obj", on _CUDA_DEVICE
│   ├── calls Mesh(verts=a float32 [3, 3] on _CUDA_DEVICE, faces=an int64 [1, 3] on _CUDA_DEVICE, texture=a MeshTextureUVTextureMap whose seam-spanning verts_uvs cross u = 1)
│   ├── calls build_texel_face_map(mesh=that mesh, texture_size=16)  # -> the texel-face map
│   ├── impls texel_face_index off that map
│   ├── impls near_one_column — the last texel column
│   ├── impls near_zero_column — the first texel column
│   ├── assert at least one near_one_column entry is face 0   # the primary copy rasterizes the right side of the seam
│   └── assert at least one near_zero_column entry is face 0  # the mirror copy rasterizes the left side of the seam
├── def test_build_texel_face_map_barycentric_recovers_face_vertex_attributes() -> None
│   ├── # barycentric-interpolating the owning face's three corner UVs (verts_uvs[faces_uvs[texel_face_index]] * texel_face_barycentric).sum(...) recovers each occupied texel's own center UV within numerical tolerance, so a corner-permuted barycentric is caught (not merely an in-range convex combination).
│   ├── calls _build_identity_uv_mesh  # -> the mesh under test
│   ├── impls texture_size — 64
│   ├── calls build_texel_face_map  # that mesh, that texture_size -> the texel-face map
│   ├── impls texel_face_index off that map
│   ├── impls texel_face_barycentric off that map
│   ├── impls occupied_mask — the texels whose texel_face_index is non-negative
│   ├── assert at least one texel is occupied
│   ├── impls corner_uvs — verts_uvs gathered at faces_uvs[texel_face_index.clamp(min=0)]
│   ├── impls interpolated_uv — corner_uvs weighted by texel_face_barycentric, summed over the corner axis
│   ├── impls axis — the per-texel center coordinate, (arange(texture_size) + 0.5) / texture_size
│   ├── impls expected_uv — that axis broadcast across columns for u and across rows for v, stacked
│   ├── impls max_error — the largest absolute interpolated-vs-expected UV difference over the occupied texels
│   └── assert max_error < 1.0e-3  # "a corner-permuted barycentric fails this", reporting max_error
├── def _install_namespace_package(package_name: str, package_path: Path) -> None
│   ├── # Installs one namespace package into sys.modules so the data tree imports without repo-level setup.
│   ├── if package_name is already in sys.modules
│   │   └── return  # already installed, nothing to do
│   ├── impls module — a fresh types.ModuleType named package_name
│   ├── impls point its __file__ at package_path / "__init__.py"
│   ├── impls point its __path__ at package_path
│   └── impls register it in sys.modules under package_name
└── def _build_identity_uv_mesh() -> Mesh
    ├── # Offers one CUDA single-face mesh whose identity UVs sit entirely inside [0, 1], the non-seam baseline the shape, top-row, and barycentric cases share.
    ├── calls MeshTextureUVTextureMap  # a 1x1x3 zero map, verts_uvs {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)}, faces_uvs [[0, 1, 2]], convention "obj", all on _CUDA_DEVICE
    ├── calls Mesh(verts=a float32 [3, 3] on _CUDA_DEVICE, faces=an int64 [1, 3] on _CUDA_DEVICE, texture=a MeshTextureUVTextureMap over a [1, 1, 3] map and a seam-safe verts_uvs)
    └── return  # that mesh
```

`tests/data/structures/three_d/mesh/test_load_save_roundtrip.py`

```text
test_load_save_roundtrip.py
├── from pathlib import Path
├── import numpy as np
├── import torch
├── from PIL import Image
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import collapse_seam_shifted_uv_rows
├── def test_load_save_obj_with_seam_face_is_byte_identical(tmp_path: Path) -> None
│   ├── # Load a hand-written seamed UV OBJ, save it back, and assert byte equality of the resulting vt / f lines — exercises the seam-shift-at-load + collapse-on-save round-trip.
│   ├── calls _write_seamed_uv_obj(directory=tmp_path / "source")  # -> source_obj_path
│   ├── calls Mesh.load(path=source_obj_path)
│   ├── impls saved_obj_path — "seam.obj" under a "saved" directory inside tmp_path
│   ├── calls mesh.save(path=saved_obj_path)
│   ├── calls _extract_lines_with_prefix(obj_path=source_obj_path, prefix="vt ")
│   ├── calls _extract_lines_with_prefix(obj_path=saved_obj_path, prefix="vt ")
│   ├── impls assert the saved vt lines equal the source's
│   ├── calls _extract_lines_with_prefix(obj_path=source_obj_path, prefix="f ")
│   ├── calls _extract_lines_with_prefix(obj_path=saved_obj_path, prefix="f ")
│   └── impls assert the saved f lines equal the source's
├── def test_load_promotes_seam_crossing_face_to_seam_safe_canonical(tmp_path: Path) -> None
│   ├── # After load, every face of a seamed mesh is non-wrapping (its largest cyclic gap over verts_uvs[faces_uvs[f]] is the wraparound gap), the seam-safe canonical form.
│   ├── calls _write_seamed_uv_obj(directory=tmp_path)
│   ├── calls Mesh.load(path=obj_path)
│   ├── impls face_corner_u — the u column of verts_uvs gathered at faces_uvs
│   └── for each face index of face_corner_u
│       ├── impls sorted_u — that face's corner u's sorted ascending
│       ├── impls interior_gaps — the successive differences of sorted_u
│       ├── impls wraparound_gap — sorted_u[0] + 1.0 - sorted_u[-1]
│       └── impls assert every interior gap is at most the wraparound gap
├── def test_save_collapses_seam_shifted_uv_rows() -> None
│   ├── # collapse_seam_shifted_uv_rows reduces canonical (U' > U) back to OBJ vt structure (U_obj == U): seam-shifted siblings at (u, v) and (u - 1, v) emit one vt line referenced by both face-corner indices.
│   ├── impls canonical_verts_uvs — a float32 [6, 2] table whose rows 4 and 5 are the seam-shifted siblings of rows 1 and 2  # impls-node-one-step:skip
│   ├── impls canonical_faces_uvs — an int64 [2, 3] table indexing one shifted face and one unshifted face                   # impls-node-one-step:skip
│   ├── calls collapse_seam_shifted_uv_rows(verts_uvs=canonical_verts_uvs, faces_uvs=canonical_faces_uvs)                    # -> obj_vt_table, obj_faces_uvs
│   ├── impls assert the returned vt table has shape (4, 2)  # U_obj == U
│   ├── for each sibling uv among (0.02, 0.25) and (0.05, 0.80)
│   │   ├── impls matches — the rows of the returned vt table close to that uv
│   │   └── impls assert exactly one row of the returned table matches that uv
│   ├── impls assert both face corners of the (0.02, 0.25) siblings reference one vt entry
│   └── impls assert both face corners of the (0.05, 0.80) siblings reference one vt entry
├── def _write_seamed_uv_obj(directory: Path) -> Path
│   ├── # Writes one hand-written seamed UV OBJ, with its sibling MTL and texture PNG, carrying one seam-crossing face and one non-seam face that share two vt rows.
│   ├── impls create directory with its parents, existing allowed
│   ├── impls obj_path, mtl_path, texture_path — "seam.obj", "seam.mtl" and "seam_texture.png" under directory  # impls-node-one-step:skip
│   ├── impls write the OBJ text — the mtllib / usemtl lines, four v lines, four vt lines whose u's straddle the wrap, and the two f lines sharing vt rows 2 and 3  # impls-node-one-step:skip
│   ├── impls write the MTL text — newmtl material0 naming the PNG as map_Kd
│   ├── calls Image.fromarray(a uniform 4x4 uint8 RGB array).save(str(texture_path))
│   └── return obj_path
└── def _extract_lines_with_prefix(obj_path: Path, prefix: str) -> str
    ├── # Extracts the lines of one OBJ file that start with a given prefix, joined with newlines in file order.
    ├── impls lines — obj_path read as utf-8 text and split on newlines  # impls-node-one-step:skip
    └── return the lines starting with prefix, joined with newlines
```
