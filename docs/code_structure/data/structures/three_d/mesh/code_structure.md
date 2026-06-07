# Mesh Data Structure Code Structure

Code-structure skeleton for `data/structures/three_d/mesh/`.

## Mesh class

`data/structures/three_d/mesh/mesh.py`

```text
mesh.py
├── from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
├── from data.structures.three_d.mesh.validate import validate_mesh_attributes
└── class Mesh
    ├── # One triangle mesh: geometry (verts, faces) plus an optional MeshTexture.
    ├── verts: torch.Tensor
    ├── faces: torch.Tensor
    ├── texture: Optional[MeshTexture]
    ├── device: torch.device
    ├── def __init__(self, verts: torch.Tensor, faces: torch.Tensor, texture: Optional[MeshTexture] = None) -> None
    │   ├── # Validates the geometry and the texture<->geometry linkage, then stores the attributes.
    │   ├── calls validate_mesh_attributes
    │   ├── impls self.verts = verts
    │   ├── impls self.faces = faces
    │   ├── impls self.texture = texture
    │   └── impls self.device = self.verts.device
    ├── @classmethod def load(cls, path: Union[str, Path]) -> "Mesh"
    │   ├── # Loads one mesh from a GLB file or an OBJ source (a single OBJ file or a mesh-root directory of OBJs).
    │   ├── from data.structures.three_d.mesh.load import load_mesh   # deferred: load.py imports mesh.py, so mesh.py must not import load.py at module level
    │   ├── calls load_mesh
    │   └── return                                  # the Mesh returned by load_mesh
    ├── def save(self, path: Union[str, Path]) -> None
    │   ├── # Saves this mesh to an OBJ/PLY file or a directory.
    │   ├── from data.structures.three_d.mesh.save import save_mesh   # deferred: save.py imports mesh.py, so mesh.py must not import save.py at module level
    │   └── calls save_mesh
    └── def to(self, device: Union[str, torch.device, None] = None, convention: Optional[str] = None) -> "Mesh"
        ├── # Returns this mesh on a target device and/or UV-origin convention (self when both already match).
        ├── calls MeshTexture.to                       # when texture is not None; delegates device + convention
        └── return Mesh                                # new Mesh wrapping the moved geometry + texture
```

## Geometry and linkage validation

`data/structures/three_d/mesh/validate.py`

```text
validate.py
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def validate_mesh_attributes(verts: torch.Tensor, faces: torch.Tensor, texture: Optional[MeshTexture] = None) -> None
│   ├── # Validates the geometry and the texture<->geometry linkage; the texture self-validates its own internal shapes.
│   ├── calls validate_verts
│   ├── calls validate_faces
│   ├── calls _validate_device_compatible
│   └── # linkage: faces index verts; MeshTextureVertexColor.vertex_color rows == V; MeshTextureUVTextureMap.faces_uvs rows == F
├── def validate_verts(obj: Any) -> None
│   └── # Validates a mesh vertex tensor (float [V,3], finite, non-empty).
├── def validate_faces(obj: Any) -> None
│   └── # Validates a mesh face tensor (integer [F,3], non-empty, non-negative indices).
└── def _validate_device_compatible(verts: torch.Tensor, faces: torch.Tensor, texture: Optional[MeshTexture]) -> None
    └── # Asserts the texture's tensors live on the verts' device.
```

## Texture: abstract base

`data/structures/three_d/mesh/texture/mesh_texture.py`

```text
mesh_texture.py
├── import abc
└── class MeshTexture(abc.ABC)
    ├── # Abstract base for a mesh's texture; concrete subclasses own the representation-specific tensors and validation.
    ├── @property @abc.abstractmethod def device(self) -> torch.device
    │   └── # The device the texture's tensors live on.
    └── @abc.abstractmethod def to(self, device: Union[str, torch.device, None] = None, convention: Optional[str] = None) -> "MeshTexture"
        └── # Returns this texture on a target device and/or UV-origin convention.
```

## Texture: vertex-color representation

`data/structures/three_d/mesh/texture/mesh_texture_vertex_color.py`

```text
mesh_texture_vertex_color.py
├── from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
├── from data.structures.three_d.mesh.texture.validate_vertex_color import validate_vertex_color
└── class MeshTextureVertexColor(MeshTexture)
    ├── # Per-vertex RGB texture: vertex_color [V,3], aligned 1:1 with the mesh's verts.
    ├── vertex_color: torch.Tensor
    ├── def __init__(self, vertex_color: torch.Tensor) -> None
    │   ├── # Validates and normalizes vertex_color, then stores it.
    │   ├── calls validate_vertex_color
    │   ├── calls MeshTextureVertexColor.normalize_vertex_color
    │   └── impls self.vertex_color = normalized vertex_color
    ├── @staticmethod def normalize_vertex_color(vertex_color: torch.Tensor) -> torch.Tensor
    │   └── # Normalizes vertex color to contiguous float32 [V,3] in [0,1] (drops a leading batch axis; uint8 -> /255).
    ├── @property def device(self) -> torch.device
    │   └── # The device vertex_color lives on.
    └── def to(self, device: Union[str, torch.device, None] = None, convention: Optional[str] = None) -> "MeshTextureVertexColor"
        ├── # Returns this texture on a target device; convention must be None (vertex color carries no UV convention).
        └── return MeshTextureVertexColor
```

## Texture: uv-texture-map representation

`data/structures/three_d/mesh/texture/mesh_texture_uv_texture_map.py`

```text
mesh_texture_uv_texture_map.py
├── from data.structures.three_d.mesh.texture.conventions import transform_verts_uvs_convention
├── from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
├── from data.structures.three_d.mesh.texture.validate_uv_texture_map import validate_uv_texture_map
└── class MeshTextureUVTextureMap(MeshTexture)
    ├── # UV-atlas texture: uv_texture_map [H,W,3] + verts_uvs [U,2] (seam-safe canonical) + faces_uvs [F,3] + UV-origin convention.
    ├── uv_texture_map: torch.Tensor
    ├── verts_uvs: torch.Tensor
    ├── faces_uvs: torch.Tensor
    ├── convention: str
    ├── def __init__(self, uv_texture_map: torch.Tensor, verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, convention: str) -> None
    │   ├── # Validates the UV representation and normalizes the texture map, then stores the attributes.
    │   ├── calls validate_uv_texture_map                  # representation-level validator (all fields + cross-field invariant)
    │   ├── calls MeshTextureUVTextureMap.normalize_uv_texture_map
    │   ├── impls self.uv_texture_map = normalized uv_texture_map
    │   ├── impls self.verts_uvs = verts_uvs
    │   ├── impls self.faces_uvs = faces_uvs
    │   └── impls self.convention = convention
    ├── @staticmethod def normalize_uv_texture_map(uv_texture_map: torch.Tensor) -> torch.Tensor
    │   └── # Normalizes the UV texture map to contiguous float32 HWC in [0,1] (drops a leading batch axis; CHW -> HWC; uint8 -> /255).
    ├── @property def device(self) -> torch.device
    │   └── # The device the UV-texture tensors live on.
    └── def to(self, device: Union[str, torch.device, None] = None, convention: Optional[str] = None) -> "MeshTextureUVTextureMap"
        ├── # Returns this texture on a target device and/or UV-origin convention.
        ├── calls transform_verts_uvs_convention        # when the convention changes
        └── return MeshTextureUVTextureMap
```

## Texture: UV-origin convention

`data/structures/three_d/mesh/texture/conventions.py`

```text
conventions.py
└── def transform_verts_uvs_convention(verts_uvs: torch.Tensor, source_convention: str, target_convention: str) -> torch.Tensor
    ├── # Transforms a UV table between origin conventions ("obj" = v from bottom, "top_left" = v from top).
    ├── if source_convention == target_convention
    │   └── return verts_uvs
    └── else
        ├── impls flipped = a copy of verts_uvs with the V axis flipped (v -> 1 - v)
        └── return flipped
```

## Texture: seam-safe canonical layout

`data/structures/three_d/mesh/texture/canonicalize.py`

```text
canonicalize.py
├── def shift_seam_crossing_faces_to_seam_safe(verts_uvs: torch.Tensor, faces_uvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Shifts seam-crossing UV faces into the seam-safe canonical chart (each face's corners made contiguous: its largest cyclic gap is the wraparound gap), forking any source vt row shared between a shifted and a non-shifted face.
│   ├── impls for each face, sort its 3 corner-u's and find the largest cyclic gap among the two interior gaps and the wraparound gap (min_u + 1 - max_u)
│   ├── impls a face is seam-crossing iff its largest cyclic gap is an INTERIOR gap (not the wraparound gap); a wide but non-wrapping face has its largest gap at the wraparound position and needs no shift
│   ├── impls for each seam-crossing face, shift the corners lying below the largest-gap cut by +1 so the cut moves to the wraparound position and the corners become contiguous (number of shifted corners is per-face, not a fixed 0.5 threshold)
│   ├── impls fork a source vt row into two when one row must be shifted by a seam-crossing face but left in place by another face sharing it (the shifted copy receives +1, the in-place copy stays)
│   └── return                                              # (verts_uvs_canonical, faces_uvs_canonical) with U' >= U
└── def collapse_seam_shifted_uv_rows(verts_uvs: torch.Tensor, faces_uvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    ├── # Collapses seam-shifted canonical UV rows back to the OBJ-style vt structure (inverse of shift_seam_crossing_faces_to_seam_safe).
    ├── impls detect canonical sibling pairs at (u, v) and (u - 1, v) within verts_uvs
    ├── impls emit one OBJ vt entry per pair and repoint both face-corner indices in faces_uvs to that entry
    ├── impls wrap u mod 1 for any canonical row without a sibling
    └── return                                              # (obj_vt_table, obj_faces_uvs) with U_obj <= U_canonical
```

## Texture: vertex-color validation

`data/structures/three_d/mesh/texture/validate_vertex_color.py`

```text
validate_vertex_color.py
├── def validate_vertex_color(obj: Any) -> None
│   ├── # Validates a vertex-color tensor ([V,3] or [1,V,3]; uint8 [0,255] or float32 [0,1]).
│   ├── if obj.dtype == torch.uint8
│   │   ├── calls _validate_vertex_color_uint8
│   │   └── return
│   ├── if obj.dtype == torch.float32
│   │   ├── calls _validate_vertex_color_float32
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _validate_vertex_color_uint8(obj: Any) -> None
│   └── # Validates a uint8 vertex-color tensor.
└── def _validate_vertex_color_float32(obj: Any) -> None
    └── # Validates a float32 vertex-color tensor (finite, RGB values within [0,1]).
```

## Texture: uv-texture-map validation

`data/structures/three_d/mesh/texture/validate_uv_texture_map.py`

```text
validate_uv_texture_map.py
├── def validate_uv_texture_map(uv_texture_map: torch.Tensor, verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, convention: str) -> None
│   ├── # Validates the whole uv-texture-map representation: every single-field validator plus the cross-field invariants.
│   ├── calls validate_uv_texture_map_image                  # single-field: uv_texture_map
│   ├── calls validate_verts_uvs                             # single-field: verts_uvs
│   ├── calls validate_faces_uvs                             # single-field: faces_uvs
│   ├── calls validate_mesh_uv_convention                    # single-field: convention
│   └── calls _validate_verts_uvs_faces_uvs_cross_field      # cross-field: (verts_uvs, faces_uvs)
├── def validate_uv_texture_map_image(obj: Any) -> None
│   ├── # Validates a UV texture image tensor (HWC/CHW/NHWC/NCHW, 3 channels; uint8 or float32).
│   ├── if obj.dtype == torch.uint8
│   │   ├── calls _validate_uv_texture_map_image_uint8
│   │   └── return
│   ├── if obj.dtype == torch.float32
│   │   ├── calls _validate_uv_texture_map_image_float32
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _validate_uv_texture_map_image_uint8(obj: Any) -> None
│   └── # Validates a uint8 UV texture image tensor.
├── def _validate_uv_texture_map_image_float32(obj: Any) -> None
│   └── # Validates a float32 UV texture image tensor (finite, values within [0,1]).
├── def validate_verts_uvs(obj: Any) -> None
│   └── # Validates a UV-coordinate table (float [U,2], finite, non-negative; values may exceed 1 — see the seam contract on MeshTextureUVTextureMap).
├── def validate_faces_uvs(obj: Any) -> None
│   └── # Validates a face-to-UV index tensor (integer [F,3], non-empty, non-negative indices).
├── def validate_mesh_uv_convention(convention: Any) -> str
│   └── # Validates and returns a UV-origin convention string (one of "obj", "top_left").
└── def _validate_verts_uvs_faces_uvs_cross_field(verts_uvs: torch.Tensor, faces_uvs: torch.Tensor) -> None
    ├── # Validates the cross-field invariants between verts_uvs and faces_uvs.
    ├── def _validate_faces_uvs_index_range() -> None [local]
    │   ├── # Asserts that every faces_uvs entry references a valid verts_uvs row.
    │   └── impls assert max(faces_uvs) < verts_uvs.shape[0]
    ├── calls _validate_faces_uvs_index_range()
    ├── def _validate_seam_safe_uv_layout() -> None [local]
    │   ├── # Asserts each face is in non-wrapping canonical form: its corners are contiguous, so its largest cyclic gap is the wraparound gap.
    │   └── impls assert per-face the wraparound gap (min_u + 1 - max_u over verts_uvs[faces_uvs[f]]) >= both interior gaps between the sorted corner-u's
    └── calls _validate_seam_safe_uv_layout()
```

## Texture: texel-to-face map

`data/structures/three_d/mesh/texture/texel_face_map.py`

```text
texel_face_map.py
├── import nvdiffrast.torch as dr
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── if TYPE_CHECKING
│   └── from data.structures.three_d.mesh.mesh import Mesh   # TYPE_CHECKING-only: avoids the mesh.py -> texture/__init__.py -> texel_face_map.py -> mesh.py import cycle
├── def build_texel_face_map(mesh: Mesh, texture_size: int) -> Dict[str, torch.Tensor]
│   ├── # Builds the texel -> mesh-face correspondence for one UV-textured mesh (requires a seam-safe canonical MeshTextureUVTextureMap) at the given texture resolution.
│   ├── calls _build_seam_safe_uv_triangle_soup(verts_uvs=mesh.texture.verts_uvs, faces=mesh.faces, faces_uvs=mesh.texture.faces_uvs)
│   ├── calls _verts_uvs_to_clip(verts_uvs=raster_verts_uvs)
│   ├── calls _compute_texel_face_index(rast_out=rast_out, raster_face_indices=raster_face_indices)
│   └── calls _compute_texel_face_barycentric(rast_out=rast_out)
├── def _build_seam_safe_uv_triangle_soup(verts_uvs: torch.Tensor, faces: torch.Tensor, faces_uvs: torch.Tensor) -> Dict[str, torch.Tensor]
│   ├── # Builds the per-face UV triangle soup, adding a u-shifted mirror copy for any face whose seam-safe corners extend outside [0, 1] so the T x T rasterizer covers both sides of the cylindrical wrap.
│   └── return                                              # "raster_verts_uvs" [Vr, 2] / "tri_i32" [Fr, 3] int32 / "raster_face_indices" [Fr] (mesh-face id per soup triangle)
├── def _verts_uvs_to_clip(verts_uvs: torch.Tensor) -> torch.Tensor
│   └── # Converts UV coordinates to clip-space positions [1, V, 4] for the UV rasterizer (u, v -> 2u - 1, 2v - 1, 0, 1).
├── def _compute_texel_face_index(rast_out: torch.Tensor, raster_face_indices: torch.Tensor) -> torch.Tensor
│   ├── # Maps the rasterizer's per-texel soup-triangle index back to the original mesh-face index.
│   ├── impls soup_triangle_index = rast_out[..., 3].long() - 1   # nvdiffrast is 1-indexed; -1 marks unoccupied texels
│   ├── impls texel_face_index = raster_face_indices[soup_triangle_index]
│   ├── impls preserve -1 sentinel where soup_triangle_index < 0
│   └── return                                              # [T, T] int64 mesh-face index map (-1 sentinel for unoccupied texels)
└── def _compute_texel_face_barycentric(rast_out: torch.Tensor) -> torch.Tensor
    ├── # Extracts per-texel face-local barycentric weights from the rasterizer output.
    ├── impls (u_bary, v_bary) = rast_out[..., 0], rast_out[..., 1]
    ├── impls (w0, w1, w2) = (1 - u_bary - v_bary, u_bary, v_bary)
    └── return                                              # [T, T, 3] barycentric weights (w0, w1, w2 summing to 1 on occupied texels)
```

## Texture: package API surface

`data/structures/three_d/mesh/texture/__init__.py`

```text
__init__.py
├── from data.structures.three_d.mesh.texture.canonicalize import collapse_seam_shifted_uv_rows, shift_seam_crossing_faces_to_seam_safe
├── from data.structures.three_d.mesh.texture.conventions import transform_verts_uvs_convention
├── from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from data.structures.three_d.mesh.texture.texel_face_map import build_texel_face_map
├── from data.structures.three_d.mesh.texture.validate_uv_texture_map import validate_uv_texture_map, validate_uv_texture_map_image, validate_verts_uvs, validate_faces_uvs, validate_mesh_uv_convention
└── from data.structures.three_d.mesh.texture.validate_vertex_color import validate_vertex_color
```

## Loading: API and format dispatch

`data/structures/three_d/mesh/load/__init__.py`

```text
__init__.py
└── from data.structures.three_d.mesh.load.load import load_mesh
```

`data/structures/three_d/mesh/load/load.py`

```text
load.py
├── from data.structures.three_d.mesh.load.load_glb import load_glb_mesh
├── from data.structures.three_d.mesh.load.load_obj import load_obj_mesh
└── def load_mesh(path: Union[str, Path]) -> Mesh
    ├── # Loads one mesh from a GLB file or an OBJ source (a single OBJ file or a mesh-root directory of OBJs).
    ├── if the path is a .glb file
    │   ├── calls load_glb_mesh
    │   └── return
    ├── if the path is an OBJ file or a mesh-root directory of OBJs
    │   ├── calls load_obj_mesh
    │   └── return
    └── assert 0, "should not reach here"
```

## Loading: OBJ

`data/structures/three_d/mesh/load/load_obj.py`

```text
load_obj.py
├── from pytorch3d.io import load_obj
├── from data.structures.three_d.mesh.load.merge import merge_meshes, pack_texture_images
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import shift_seam_crossing_faces_to_seam_safe
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def load_obj_mesh(path: Union[str, Path]) -> Mesh
│   ├── # Loads one OBJ file, or every OBJ under a mesh-root directory, into one merged Mesh (single or multiple blocks).
│   ├── calls _resolve_input_paths
│   ├── calls _load_mesh_block_from_obj_path                  # per OBJ block
│   └── calls merge_meshes
├── def _load_mesh_block_from_obj_path(obj_path: Path) -> Mesh
│   ├── # Loads one OBJ as a single mesh block, dispatched to the texture-representation-specific loader.
│   ├── calls _inspect_obj_file
│   ├── if not has_vertex_colors and not (has_uv_coords and has_uv_faces)
│   │   ├── calls _load_mesh_geometry_only
│   │   └── return
│   ├── if has_vertex_colors
│   │   ├── calls _load_mesh_vertex_color
│   │   └── return
│   ├── if has_uv_coords and has_uv_faces
│   │   ├── calls _load_mesh_uv_texture_map
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _load_mesh_geometry_only(path: Union[str, Path]) -> Mesh
│   ├── # Loads a geometry-only OBJ (parses v / f lines; texture None).
│   └── calls _resolve_input_path
├── def _load_mesh_vertex_color(path: Union[str, Path]) -> Mesh
│   ├── # Loads a vertex-colored OBJ (parses v-with-RGB / f lines) into a MeshTextureVertexColor-textured mesh.
│   └── calls _resolve_input_path
├── def _load_mesh_uv_texture_map(path: Union[str, Path]) -> Mesh
│   ├── # Loads a UV-textured OBJ via PyTorch3D into a MeshTextureUVTextureMap-textured mesh on the geometry domain (convention "obj").
│   ├── calls _resolve_input_path
│   ├── calls load_obj                                        # verts, faces, aux (verts_uvs, textures_idx, texture_images)
│   ├── calls pack_texture_images                             # multi-material -> single atlas
│   └── calls shift_seam_crossing_faces_to_seam_safe          # raw OBJ verts_uvs -> seam-safe canonical
├── def _resolve_input_path(path: Union[str, Path]) -> Path
│   ├── # Resolves a mesh path to exactly one OBJ file.
│   └── calls _resolve_input_paths
├── def _resolve_input_paths(path: Union[str, Path]) -> List[Path]
│   └── # Resolves a mesh path to one OBJ file, or every OBJ at the top level / one level below a directory.
└── def _inspect_obj_file(obj_path: Path) -> Dict[str, bool]
    └── # Inspects one OBJ to detect its texture representation (has_vertex_colors / has_uv_coords / has_uv_faces / has_mtllib).
```

## Loading: GLB

`data/structures/three_d/mesh/load/load_glb.py`

```text
load_glb.py
├── import numpy as np
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import shift_seam_crossing_faces_to_seam_safe
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from utils.io.glb import load_glb_json_and_bin, read_accessor, read_image_bytes
├── from utils.io.image import decode_image_bytes
├── def load_glb_mesh(path: Union[str, Path]) -> Mesh
│   ├── # Loads one GLB file into a Mesh, dispatched to the texture-representation-specific loader (a GLB is one self-contained file, so no block split).
│   ├── calls load_glb_json_and_bin
│   ├── calls _select_mesh_primitive
│   ├── if the primitive has neither a COLOR_0 attribute nor a TEXCOORD_0 with a base-color texture
│   │   ├── calls _load_glb_geometry_only
│   │   └── return
│   ├── if the primitive has a COLOR_0 attribute
│   │   ├── calls _load_glb_vertex_color
│   │   └── return
│   ├── if the primitive has TEXCOORD_0 and a base-color texture
│   │   ├── calls _load_glb_uv_texture_map
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _load_glb_geometry_only(gltf: Dict[str, Any], binary_blob: bytes, mesh_index: int, primitive_index: int) -> Mesh
│   ├── # Builds a geometry-only Mesh from a GLB primitive (POSITION -> verts; indices -> faces; texture None).
│   ├── calls read_accessor                                  # POSITION -> verts, indices -> faces
│   └── return Mesh                                           # texture None
├── def _load_glb_vertex_color(gltf: Dict[str, Any], binary_blob: bytes, mesh_index: int, primitive_index: int) -> Mesh
│   ├── # Builds a vertex-colored Mesh from a GLB primitive (POSITION -> verts; indices -> faces; COLOR_0 -> vertex_color).
│   ├── calls read_accessor                                  # POSITION -> verts, indices -> faces, COLOR_0 -> vertex_color
│   └── return Mesh                                           # MeshTextureVertexColor
├── def _load_glb_uv_texture_map(gltf: Dict[str, Any], binary_blob: bytes, mesh_index: int, primitive_index: int) -> Mesh
│   ├── # Builds a UV-textured Mesh from a GLB primitive (POSITION -> verts; the shared index buffer -> faces and the raw faces_uvs; TEXCOORD_0 -> verts_uvs; base-color image -> uv_texture_map) on convention "top_left".
│   ├── calls read_accessor                                  # POSITION -> verts, indices -> faces, TEXCOORD_0 -> verts_uvs
│   ├── calls _resolve_base_color_texture_image_index
│   ├── calls read_image_bytes
│   ├── calls decode_image_bytes
│   ├── calls shift_seam_crossing_faces_to_seam_safe          # raw glTF verts_uvs (faces_uvs = the shared index buffer) -> seam-safe canonical
│   └── return Mesh                                           # MeshTextureUVTextureMap(convention="top_left")
├── def _select_mesh_primitive(gltf: Dict[str, Any]) -> Tuple[int, int]
│   └── # Selects the (mesh_index, primitive_index) to load — the primitive whose material carries a base-color texture (for a GLB of many untextured marker meshes plus one textured face, this uniquely picks the face).
└── def _resolve_base_color_texture_image_index(gltf: Dict[str, Any], primitive: Dict[str, Any]) -> int
    └── # Resolves a primitive's material base-color texture to its glTF image index (glTF-semantic material navigation).
```

## Loading: block merging

`data/structures/three_d/mesh/load/merge.py`

```text
merge.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def merge_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   ├── # Merges one or more mesh blocks (assumed homogeneous in texture representation) into one Mesh.
│   ├── if len(mesh_blocks) == 1
│   │   └── return mesh_blocks[0]                             # single block: pass-through
│   ├── if no block carries a texture
│   │   ├── calls _merge_geometry_only_meshes
│   │   └── return
│   ├── if any block carries MeshTextureVertexColor
│   │   ├── calls _merge_vertex_color_meshes
│   │   └── return
│   ├── if any block carries MeshTextureUVTextureMap
│   │   ├── calls _merge_uv_textured_meshes
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _merge_geometry_only_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   └── # Merges geometry-only mesh blocks, concatenating geometry with vertex offsets.
├── def _merge_vertex_color_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   └── # Merges vertex-colored mesh blocks, concatenating geometry and vertex colors with vertex offsets.
├── def _merge_uv_textured_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   ├── # Merges UV-textured mesh blocks, concatenating geometry and UV and packing per-block textures into one atlas.
│   └── calls _pack_texture_maps
├── def pack_texture_images(texture_images: Dict[str, torch.Tensor], verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, materials_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Packs a material-name -> image mapping into one atlas plus remapped UVs.
│   └── calls _pack_texture_maps
├── def _pack_texture_maps(texture_maps: Sequence[torch.Tensor], verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, materials_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Stacks texture maps into one atlas and rebuilds the per-corner UV table.
│   └── calls _remap_uvs
└── def _remap_uvs(verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, map_offsets: torch.Tensor, atlas_height: int, atlas_width: int, materials_idx: torch.Tensor) -> torch.Tensor
    └── # Rescales and offsets each material's UVs into its packed atlas region.
```

## Saving: API and format dispatch

`data/structures/three_d/mesh/save/__init__.py`

```text
__init__.py
└── from data.structures.three_d.mesh.save.save import save_mesh
```

`data/structures/three_d/mesh/save/save.py`

```text
save.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.save.save_glb import save_glb_mesh
├── from data.structures.three_d.mesh.save.save_obj import save_obj_mesh
├── from data.structures.three_d.mesh.save.save_ply import save_ply_mesh
└── def save_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None
    ├── # Saves a mesh to OBJ, PLY, or GLB, dispatched on the output path's format (OBJ vs PLY vs GLB is the top-level responsibility split; a directory defaults to an OBJ file).
    ├── if the output path is a .glb file
    │   ├── calls save_glb_mesh
    │   └── return
    ├── if the output path is a .ply file
    │   ├── calls save_ply_mesh
    │   └── return
    ├── if the output path is an .obj file or a directory
    │   ├── calls save_obj_mesh
    │   └── return
    └── assert 0, "should not reach here"
```

## Saving: OBJ

`data/structures/three_d/mesh/save/save_obj.py`

```text
save_obj.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import collapse_seam_shifted_uv_rows
├── from data.structures.three_d.mesh.texture.conventions import transform_verts_uvs_convention
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def save_obj_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Writes a Mesh to OBJ, dispatched to the texture-representation-specific writer.
│   ├── calls _resolve_output_obj_path
│   ├── if mesh.texture is None
│   │   ├── calls _save_geometry_only_obj
│   │   └── return
│   ├── if isinstance(mesh.texture, MeshTextureVertexColor)
│   │   ├── calls _save_vertex_color_obj
│   │   └── return
│   ├── if isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   ├── calls _save_uv_texture_map_obj
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _save_geometry_only_obj(mesh: Mesh, obj_path: Path) -> None
│   └── # Writes the OBJ v / f lines.
├── def _save_vertex_color_obj(mesh: Mesh, obj_path: Path) -> None
│   ├── # Writes the OBJ v-x-y-z-r-g-b / f lines.
│   └── calls _normalize_vertex_color_for_obj
├── def _save_uv_texture_map_obj(mesh: Mesh, obj_path: Path) -> None
│   ├── # Writes the OBJ plus a sibling MTL and texture PNG.
│   ├── calls _normalize_uv_texture_map_for_png
│   ├── calls transform_verts_uvs_convention                  # texture convention -> "obj" for the written vt lines
│   └── calls collapse_seam_shifted_uv_rows                   # seam-safe canonical -> OBJ vt structure
├── def _resolve_output_obj_path(output_path: Union[str, Path]) -> Path
│   └── # Resolves an output path to a concrete .obj file path (an ".obj" path, or "<dir>/mesh.obj").
├── def _normalize_vertex_color_for_obj(vertex_color: torch.Tensor) -> torch.Tensor
│   └── # Normalizes vertex color to float32 [0,1] for OBJ export.
└── def _normalize_uv_texture_map_for_png(uv_texture_map: torch.Tensor) -> np.ndarray
    └── # Normalizes a UV texture map to a uint8 HWC array for PNG export.
```

## Saving: PLY

`data/structures/three_d/mesh/save/save_ply.py`

```text
save_ply.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def save_ply_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Writes a Mesh to PLY, dispatched to the texture-representation-specific writer (PLY carries geometry + optional per-vertex color; a UV-atlas texture has no PLY representation).
│   ├── calls _resolve_output_ply_path
│   ├── if mesh.texture is None
│   │   ├── calls _save_geometry_only_ply
│   │   └── return
│   ├── if isinstance(mesh.texture, MeshTextureVertexColor)
│   │   ├── calls _save_vertex_color_ply
│   │   └── return
│   ├── if isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   └── raise ValueError                                  # a UV-atlas texture cannot be written to PLY; save to OBJ or GLB
│   └── assert 0, "should not reach here"
├── def _save_geometry_only_ply(mesh: Mesh, ply_path: Path) -> None
│   └── # Writes a geometry-only PLY.
├── def _save_vertex_color_ply(mesh: Mesh, ply_path: Path) -> None
│   ├── # Writes a vertex-colored PLY.
│   └── calls _normalize_vertex_color_for_ply
├── def _resolve_output_ply_path(output_path: Union[str, Path]) -> Path
│   └── # Resolves an output path to a concrete .ply file path (a ".ply" path, or "<dir>/mesh.ply").
└── def _normalize_vertex_color_for_ply(vertex_color: torch.Tensor) -> torch.Tensor
    └── # Normalizes vertex color to uint8 [0,255] for PLY export.
```

## Saving: GLB

`data/structures/three_d/mesh/save/save_glb.py`

```text
save_glb.py
├── import numpy as np
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import collapse_seam_shifted_uv_rows
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from utils.io.glb import append_accessor, append_image, write_glb
├── from utils.io.image import encode_image_bytes
├── def save_glb_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Writes a Mesh to a GLB container, dispatched to the texture-representation-specific writer.
│   ├── calls _resolve_output_glb_path
│   ├── if mesh.texture is None
│   │   ├── calls _save_geometry_only_glb
│   │   └── return
│   ├── if isinstance(mesh.texture, MeshTextureVertexColor)
│   │   ├── calls _save_vertex_color_glb
│   │   └── return
│   ├── if isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   ├── calls _save_uv_texture_map_glb
│   │   └── return
│   └── assert 0, "should not reach here"
├── def _save_geometry_only_glb(mesh: Mesh, glb_path: Path) -> None
│   ├── # Appends POSITION + indices accessors and writes the GLB (no material).
│   ├── calls append_accessor
│   └── calls write_glb
├── def _save_vertex_color_glb(mesh: Mesh, glb_path: Path) -> None
│   ├── # Appends POSITION + indices + COLOR_0 accessors and writes the GLB (no texture material).
│   ├── calls append_accessor
│   └── calls write_glb
├── def _save_uv_texture_map_glb(mesh: Mesh, glb_path: Path) -> None
│   ├── # Appends POSITION + indices + TEXCOORD_0 accessors + an embedded base-color texture image, then writes the GLB.
│   ├── calls collapse_seam_shifted_uv_rows                   # seam-safe canonical -> glTF shared-index vt structure
│   ├── calls append_accessor
│   ├── calls encode_image_bytes
│   ├── calls append_image
│   └── calls write_glb
└── def _resolve_output_glb_path(output_path: Union[str, Path]) -> Path
    └── # Resolves an output path to a concrete .glb file path (a ".glb" path, or "<dir>/mesh.glb").
```

## Framework interop conversions

`data/structures/three_d/mesh/convert.py`

```text
convert.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import collapse_seam_shifted_uv_rows, shift_seam_crossing_faces_to_seam_safe
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def mesh_from_open3d(mesh: o3d.geometry.TriangleMesh) -> Mesh
│   └── # Converts an Open3D triangle mesh into a Mesh (geometry plus optional MeshTextureVertexColor; UV not supported).
├── def mesh_to_open3d(mesh: Mesh) -> o3d.geometry.TriangleMesh
│   ├── # Converts a Mesh into an Open3D triangle mesh (geometry plus optional vertex colors; UV not supported).
│   └── calls _vertex_color_to_float_rgb
├── def mesh_from_pytorch3d(mesh: Meshes, convention: str = "obj") -> Mesh
│   ├── # Converts a PyTorch3D Meshes into a Mesh.
│   ├── if mesh.textures is None
│   │   └── # builds a geometry-only Mesh
│   ├── elif isinstance(mesh.textures, TexturesVertex)
│   │   └── # builds Mesh with a MeshTextureVertexColor
│   └── else  # TexturesUV
│       ├── calls shift_seam_crossing_faces_to_seam_safe     # raw TexturesUV verts_uvs -> seam-safe canonical
│       └── # builds Mesh with a MeshTextureUVTextureMap
├── def mesh_to_pytorch3d(mesh: Mesh, device: Union[str, torch.device, None] = None, dtype: torch.dtype = torch.float32) -> Meshes
│   ├── # Converts a Mesh into a PyTorch3D Meshes.
│   ├── if isinstance(mesh.texture, MeshTextureVertexColor)
│   │   └── # builds Meshes with a TexturesVertex
│   ├── elif isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   ├── calls collapse_seam_shifted_uv_rows              # seam-safe canonical -> OBJ vt structure for TexturesUV
│   │   └── # builds Meshes with a TexturesUV (UV forced to "obj" convention)
│   └── else
│       └── # builds a geometry-only Meshes
├── def mesh_from_trimesh(mesh: trimesh.Trimesh, convention: Optional[str] = None) -> Mesh
│   ├── # Converts a trimesh.Trimesh into a Mesh.
│   ├── if mesh.visual carries uv
│   │   ├── calls _uv_mesh_from_trimesh                      # welds per-corner duplicate verts into the geometry domain
│   │   ├── calls shift_seam_crossing_faces_to_seam_safe     # raw trimesh verts_uvs -> seam-safe canonical
│   │   ├── calls _texture_image_from_trimesh
│   │   └── # builds Mesh with a MeshTextureUVTextureMap
│   └── else
│       ├── calls _vertex_color_from_trimesh
│       └── # builds Mesh with a MeshTextureVertexColor
├── def mesh_to_trimesh(mesh: Mesh) -> trimesh.Trimesh
│   ├── # Converts a Mesh into a trimesh.Trimesh.
│   ├── if isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   ├── calls _uv_mesh_to_trimesh                        # expands to per-corner topology
│   │   └── calls _texture_image_to_trimesh
│   ├── elif isinstance(mesh.texture, MeshTextureVertexColor)
│   │   └── calls _vertex_color_to_trimesh
│   └── else
│       └── # geometry-only Trimesh
├── def _uv_mesh_from_trimesh(verts: np.ndarray, faces: np.ndarray, verts_uvs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   └── # Welds trimesh's per-corner duplicate verts (exact-position equality) into the geometry domain, returning (verts, faces, verts_uvs, faces_uvs).
├── def _texture_image_from_trimesh(image: object) -> np.ndarray
│   └── # Converts a trimesh material image to a uint8 HWC RGB array (drops uniform alpha).
├── def _vertex_color_from_trimesh(vertex_colors: np.ndarray) -> np.ndarray
│   └── # Converts trimesh vertex colors to a repo RGB array (drops opaque alpha).
├── def _uv_mesh_to_trimesh(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
│   ├── # Expands an "obj"-convention UV mesh to trimesh's per-corner topology, returning (verts, faces, uv).
│   └── calls collapse_seam_shifted_uv_rows                  # seam-safe canonical -> OBJ vt structure before per-corner expansion
├── def _texture_image_to_trimesh(uv_texture_map: torch.Tensor) -> np.ndarray
│   └── # Converts a repo uv_texture_map tensor to a uint8 HWC RGB array.
├── def _vertex_color_to_trimesh(vertex_color: torch.Tensor) -> np.ndarray
│   ├── # Converts a repo vertex_color tensor to a uint8 RGBA array for trimesh.
│   └── calls _vertex_color_to_float_rgb
└── def _vertex_color_to_float_rgb(vertex_color: torch.Tensor) -> np.ndarray
    └── # Converts a repo vertex_color tensor to a float32 RGB [0,1] array; shared by mesh_to_open3d and _vertex_color_to_trimesh.
```

## Package API surface

`data/structures/three_d/mesh/__init__.py`

```text
__init__.py
├── from data.structures.three_d.mesh import texture
├── from data.structures.three_d.mesh.convert import mesh_from_open3d, mesh_from_pytorch3d, mesh_from_trimesh, mesh_to_open3d, mesh_to_pytorch3d, mesh_to_trimesh
├── from data.structures.three_d.mesh.load import load_mesh
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.save import save_mesh
└── from data.structures.three_d.mesh.validate import validate_faces, validate_mesh_attributes, validate_verts
```
