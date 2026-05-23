# Mesh Data Structure Code Structure

Code-structure skeleton for `data/structures/three_d/mesh/`.

## Mesh class

```text
data/structures/three_d/mesh/mesh.py
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
    │   ├── # Loads one mesh from an OBJ file or a mesh-root directory.
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

```text
data/structures/three_d/mesh/validate.py
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def validate_verts(obj: Any) -> None
│   └── # Validates a mesh vertex tensor (float [V,3], finite, non-empty).
├── def validate_faces(obj: Any) -> None
│   └── # Validates a mesh face tensor (integer [F,3], non-empty, non-negative indices).
├── def _validate_device_compatible(verts: torch.Tensor, faces: torch.Tensor, texture: Optional[MeshTexture]) -> None
│   └── # Asserts the texture's tensors live on the verts' device.
└── def validate_mesh_attributes(verts: torch.Tensor, faces: torch.Tensor, texture: Optional[MeshTexture] = None) -> None
    ├── # Validates the geometry and the texture<->geometry linkage; the texture self-validates its own internal shapes.
    ├── calls validate_verts
    ├── calls validate_faces
    ├── calls _validate_device_compatible
    └── # linkage: faces index verts; MeshTextureVertexColor.vertex_color rows == V; MeshTextureUVTextureMap.faces_uvs rows == F
```

## Texture: abstract base

```text
data/structures/three_d/mesh/texture/mesh_texture.py
├── import abc
└── class MeshTexture(abc.ABC)
    ├── # Abstract base for a mesh's texture; concrete subclasses own the representation-specific tensors and validation.
    ├── @property @abc.abstractmethod def device(self) -> torch.device
    │   └── # The device the texture's tensors live on.
    └── @abc.abstractmethod def to(self, device: Union[str, torch.device, None] = None, convention: Optional[str] = None) -> "MeshTexture"
        └── # Returns this texture on a target device and/or UV-origin convention.
```

## Texture: vertex-color representation

```text
data/structures/three_d/mesh/texture/mesh_texture_vertex_color.py
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

```text
data/structures/three_d/mesh/texture/mesh_texture_uv_texture_map.py
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

```text
data/structures/three_d/mesh/texture/conventions.py
└── def transform_verts_uvs_convention(verts_uvs: torch.Tensor, source_convention: str, target_convention: str) -> torch.Tensor
    ├── # Transforms a UV table between origin conventions ("obj" = v from bottom, "top_left" = v from top).
    ├── if source_convention == target_convention
    │   └── return verts_uvs
    └── else
        ├── impls flipped = a copy of verts_uvs with the V axis flipped (v -> 1 - v)
        └── return flipped
```

## Texture: vertex-color validation

```text
data/structures/three_d/mesh/texture/validate_vertex_color.py
├── def validate_vertex_color(obj: Any) -> None
│   ├── # Validates a vertex-color tensor ([V,3] or [1,V,3]; uint8 [0,255] or float32 [0,1]).
│   ├── assert obj.dtype in (torch.uint8, torch.float32)      # supported dtype gate
│   ├── if obj.dtype == torch.uint8
│   │   └── calls _validate_vertex_color_uint8
│   └── if obj.dtype == torch.float32
│       └── calls _validate_vertex_color_float32
├── def _validate_vertex_color_uint8(obj: Any) -> None
│   └── # Validates a uint8 vertex-color tensor.
└── def _validate_vertex_color_float32(obj: Any) -> None
    └── # Validates a float32 vertex-color tensor (finite, RGB values within [0,1]).
```

## Texture: uv-texture-map validation

```text
data/structures/three_d/mesh/texture/validate_uv_texture_map.py
├── def validate_uv_texture_map(uv_texture_map: torch.Tensor, verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, convention: str) -> None
│   ├── # Validates the whole uv-texture-map representation: every single-field validator plus the cross-field invariants.
│   ├── calls validate_uv_texture_map_image                  # single-field: uv_texture_map
│   ├── calls validate_verts_uvs                             # single-field: verts_uvs
│   ├── calls validate_faces_uvs                             # single-field: faces_uvs
│   ├── calls validate_mesh_uv_convention                    # single-field: convention
│   └── calls _validate_verts_uvs_faces_uvs_cross_field      # cross-field: (verts_uvs, faces_uvs)
├── def validate_uv_texture_map_image(obj: Any) -> None
│   ├── # Validates a UV texture image tensor (HWC/CHW/NHWC/NCHW, 3 channels; uint8 or float32).
│   ├── assert obj.dtype in (torch.uint8, torch.float32)      # supported dtype gate
│   ├── if obj.dtype == torch.uint8
│   │   └── calls _validate_uv_texture_map_image_uint8
│   └── if obj.dtype == torch.float32
│       └── calls _validate_uv_texture_map_image_float32
├── def validate_verts_uvs(obj: Any) -> None
│   └── # Validates a UV-coordinate table (float [U,2], finite, non-negative; values may exceed 1 — see the seam contract on MeshTextureUVTextureMap).
├── def validate_faces_uvs(obj: Any) -> None
│   └── # Validates a face-to-UV index tensor (integer [F,3], non-empty, non-negative indices).
├── def validate_mesh_uv_convention(convention: Any) -> str
│   └── # Validates and returns a UV-origin convention string (one of "obj", "top_left").
├── def _validate_uv_texture_map_image_uint8(obj: Any) -> None
│   └── # Validates a uint8 UV texture image tensor.
└── def _validate_uv_texture_map_image_float32(obj: Any) -> None
    └── # Validates a float32 UV texture image tensor (finite, values within [0,1]).
```

## Texture: package API surface

```text
data/structures/three_d/mesh/texture/__init__.py
├── from data.structures.three_d.mesh.texture.conventions import transform_verts_uvs_convention
├── from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from data.structures.three_d.mesh.texture.validate_uv_texture_map import validate_uv_texture_map, validate_uv_texture_map_image, validate_verts_uvs, validate_faces_uvs, validate_mesh_uv_convention
└── from data.structures.three_d.mesh.texture.validate_vertex_color import validate_vertex_color
```

## Loading

```text
data/structures/three_d/mesh/load.py
├── from pytorch3d.io import load_obj
├── from data.structures.three_d.mesh.merge import merge_meshes, pack_texture_images
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def load_mesh(path: Union[str, Path]) -> Mesh
│   ├── # Loads one OBJ file, or every OBJ under a mesh-root directory, into one merged Mesh.
│   ├── calls _resolve_input_paths
│   ├── calls _load_mesh_block_from_obj_path                  # per OBJ block
│   └── calls merge_meshes
├── def _load_mesh_block_from_obj_path(obj_path: Path) -> Mesh
│   ├── # Loads one OBJ as a single mesh block, dispatched to the texture-representation-specific loader.
│   ├── calls _inspect_obj_file
│   ├── if has_uv_coords and has_uv_faces
│   │   └── calls _load_mesh_uv_texture_map
│   ├── elif has_vertex_colors
│   │   └── calls _load_mesh_vertex_color
│   └── else
│       └── calls _load_mesh_geometry_only
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
│   └── calls pack_texture_images                             # multi-material -> single atlas
├── def _resolve_input_path(path: Union[str, Path]) -> Path
│   ├── # Resolves a mesh path to exactly one OBJ file.
│   └── calls _resolve_input_paths
├── def _resolve_input_paths(path: Union[str, Path]) -> List[Path]
│   └── # Resolves a mesh path to one OBJ file, or every OBJ at the top level / one level below a directory.
└── def _inspect_obj_file(obj_path: Path) -> Dict[str, bool]
    └── # Inspects one OBJ to detect its texture representation (has_vertex_colors / has_uv_coords / has_uv_faces / has_mtllib).
```

## Saving

```text
data/structures/three_d/mesh/save.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.conventions import transform_verts_uvs_convention
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def save_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Dispatches on the mesh's texture type to the matching writer.
│   ├── if isinstance(mesh.texture, MeshTextureVertexColor)
│   │   └── calls _save_mesh_vertex_color
│   ├── elif isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   └── calls _save_mesh_uv_texture_map
│   └── else
│       └── calls _save_mesh_geometry_only                    # texture is None
├── def _save_mesh_geometry_only(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Writes a geometry-only mesh as OBJ or PLY.
│   ├── calls _resolve_output_non_uv_mesh_path
│   ├── if the resolved path is an .obj
│   │   └── # writes the OBJ (v / f lines)
│   └── else  # .ply
│       └── # writes the PLY
├── def _save_mesh_vertex_color(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Writes a vertex-colored mesh as OBJ (v x y z r g b) or PLY.
│   ├── calls _resolve_output_non_uv_mesh_path
│   ├── if the resolved path is an .obj
│   │   └── calls _normalize_vertex_color_for_obj
│   └── else  # .ply
│       └── calls _normalize_vertex_color_for_ply
├── def _save_mesh_uv_texture_map(mesh: Mesh, output_path: Union[str, Path]) -> None
│   ├── # Writes a UV-textured mesh as an OBJ plus a sibling MTL and texture PNG.
│   ├── calls _resolve_output_obj_path
│   ├── calls _normalize_uv_texture_map_for_png
│   └── calls transform_verts_uvs_convention                  # texture convention -> "obj" for the written vt lines
├── def _resolve_output_obj_path(output_path: Union[str, Path]) -> Path
│   └── # Resolves an output path to a concrete .obj file path (an ".obj" path, or "<dir>/mesh.obj").
├── def _resolve_output_non_uv_mesh_path(output_path: Union[str, Path]) -> Path
│   └── # Resolves an output path to a concrete .obj or .ply file path (an ".obj"/".ply" path, or "<dir>/mesh.obj").
├── def _normalize_vertex_color_for_obj(vertex_color: torch.Tensor) -> torch.Tensor
│   └── # Normalizes vertex color to float32 [0,1] for OBJ export.
├── def _normalize_vertex_color_for_ply(vertex_color: torch.Tensor) -> torch.Tensor
│   └── # Normalizes vertex color to uint8 [0,255] for PLY export.
└── def _normalize_uv_texture_map_for_png(uv_texture_map: torch.Tensor) -> np.ndarray
    └── # Normalizes a UV texture map to a uint8 HWC array for PNG export.
```

## Merging and texture-atlas packing

```text
data/structures/three_d/mesh/merge.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def merge_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   ├── # Merges one or more mesh blocks into one Mesh.
│   ├── if len(mesh_blocks) == 1
│   │   └── return mesh_blocks[0]                             # single block: pass-through
│   ├── if any block carries MeshTextureUVTextureMap
│   │   └── calls _merge_uv_textured_meshes
│   ├── elif any block carries MeshTextureVertexColor
│   │   └── calls _merge_vertex_color_meshes
│   └── else
│       └── calls _merge_geometry_only_meshes
├── def _merge_vertex_color_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   └── # Merges vertex-colored mesh blocks, concatenating geometry and vertex colors with vertex offsets.
├── def _merge_uv_textured_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   ├── # Merges UV-textured mesh blocks, concatenating geometry and UV and packing per-block textures into one atlas.
│   └── calls _pack_texture_maps
├── def _merge_geometry_only_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh
│   └── # Merges geometry-only mesh blocks, concatenating geometry with vertex offsets.
├── def pack_texture_images(texture_images: Dict[str, torch.Tensor], verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, materials_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Packs a material-name -> image mapping into one atlas plus remapped UVs.
│   └── calls _pack_texture_maps
├── def _pack_texture_maps(texture_maps: Sequence[torch.Tensor], verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, materials_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Stacks texture maps into one atlas and rebuilds the per-corner UV table.
│   └── calls _remap_uvs
└── def _remap_uvs(verts_uvs: torch.Tensor, faces_uvs: torch.Tensor, map_offsets: torch.Tensor, atlas_height: int, atlas_width: int, materials_idx: torch.Tensor) -> torch.Tensor
    └── # Rescales and offsets each material's UVs into its packed atlas region.
```

## Framework interop conversions

```text
data/structures/three_d/mesh/convert.py
├── from data.structures.three_d.mesh.mesh import Mesh
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
│       └── # builds Mesh with a MeshTextureUVTextureMap
├── def mesh_to_pytorch3d(mesh: Mesh, device: Union[str, torch.device, None] = None, dtype: torch.dtype = torch.float32) -> Meshes
│   ├── # Converts a Mesh into a PyTorch3D Meshes.
│   ├── if isinstance(mesh.texture, MeshTextureVertexColor)
│   │   └── # builds Meshes with a TexturesVertex
│   ├── elif isinstance(mesh.texture, MeshTextureUVTextureMap)
│   │   └── # builds Meshes with a TexturesUV (UV forced to "obj" convention)
│   └── else
│       └── # builds a geometry-only Meshes
├── def mesh_from_trimesh(mesh: trimesh.Trimesh, convention: Optional[str] = None) -> Mesh
│   ├── # Converts a trimesh.Trimesh into a Mesh.
│   ├── if mesh.visual carries uv
│   │   ├── calls _uv_mesh_from_trimesh                      # welds per-corner duplicate verts into the geometry domain
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
├── def _vertex_color_to_float_rgb(vertex_color: torch.Tensor) -> np.ndarray
│   └── # Converts a repo vertex_color tensor to a float32 RGB [0,1] array; shared by mesh_to_open3d and _vertex_color_to_trimesh.
├── def _uv_mesh_from_trimesh(verts: np.ndarray, faces: np.ndarray, verts_uvs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   └── # Welds trimesh's per-corner duplicate verts (exact-position equality) into the geometry domain, returning (verts, faces, verts_uvs, faces_uvs).
├── def _texture_image_from_trimesh(image: object) -> np.ndarray
│   └── # Converts a trimesh material image to a uint8 HWC RGB array (drops uniform alpha).
├── def _vertex_color_from_trimesh(vertex_colors: np.ndarray) -> np.ndarray
│   └── # Converts trimesh vertex colors to a repo RGB array (drops opaque alpha).
├── def _uv_mesh_to_trimesh(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
│   └── # Expands an "obj"-convention UV mesh to trimesh's per-corner topology, returning (verts, faces, uv).
├── def _texture_image_to_trimesh(uv_texture_map: torch.Tensor) -> np.ndarray
│   └── # Converts a repo uv_texture_map tensor to a uint8 HWC RGB array.
└── def _vertex_color_to_trimesh(vertex_color: torch.Tensor) -> np.ndarray
    ├── # Converts a repo vertex_color tensor to a uint8 RGBA array for trimesh.
    └── calls _vertex_color_to_float_rgb
```

## Package API surface

```text
data/structures/three_d/mesh/__init__.py
├── from data.structures.three_d.mesh import texture
├── from data.structures.three_d.mesh.convert import mesh_from_open3d, mesh_from_pytorch3d, mesh_from_trimesh, mesh_to_open3d, mesh_to_pytorch3d, mesh_to_trimesh
├── from data.structures.three_d.mesh.load import load_mesh
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.merge import merge_meshes
├── from data.structures.three_d.mesh.save import save_mesh
└── from data.structures.three_d.mesh.validate import validate_faces, validate_mesh_attributes, validate_verts
```
