# Mesh Data Structure Code Structure

Code-structure skeleton for `data/structures/three_d/mesh/`.

## Mesh class

```text
data/structures/three_d/mesh/mesh.py
├── from data.structures.three_d.mesh.load import load_mesh
├── from data.structures.three_d.mesh.save import save_mesh
├── from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
├── from data.structures.three_d.mesh.validate import validate_mesh_attributes
└── class Mesh
    ├── # One triangle mesh: geometry (vertices, faces) plus an optional MeshTexture.
    ├── def __init__(self, vertices, faces, texture=None)
    │   ├── # Validates geometry + the texture<->geometry linkage, then stores the attributes.
    │   ├── calls validate_mesh_attributes
    │   ├── impls self.vertices = vertices, contiguous
    │   ├── impls self.faces = faces, int64 + contiguous
    │   ├── impls self.texture = texture                # Optional[MeshTexture]
    │   └── impls self.device = self.vertices.device
    ├── @classmethod def load(cls, path) -> Mesh
    │   ├── # Loads one mesh from an OBJ file or a mesh-root directory.
    │   └── calls load_mesh
    ├── def save(self, path) -> None
    │   ├── # Saves this mesh to an OBJ/PLY file or a directory.
    │   └── calls save_mesh
    └── def to(self, device=None, convention=None) -> Mesh
        ├── # Returns this mesh on a target device and/or UV-origin convention; self when both already match.
        ├── calls MeshTexture.to                       # when texture is not None; delegates device + convention
        └── return Mesh                                # new Mesh wrapping the moved geometry + texture
```

## Geometry and linkage validation

```text
data/structures/three_d/mesh/validate.py
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def validate_vertices(obj) -> None                        # float [V,3], V>0, finite
├── def validate_faces(obj) -> None                           # integer [F,3], F>0, indices >= 0
├── def _validate_device_compatible(vertices, faces, texture) -> None
│   └── # Asserts the texture's tensors live on the vertices' device.
└── def validate_mesh_attributes(vertices, faces, texture=None) -> None
    ├── # Validates geometry plus the texture<->geometry linkage (the texture self-validates its own internal shapes).
    ├── calls validate_vertices
    ├── calls validate_faces
    ├── calls _validate_device_compatible
    ├── calls validate_vertex_color           # when vertex_color is set
    ├── calls validate_vertex_uv / validate_face_uvs / validate_mesh_uv_convention   # when UV topology is set
    ├── calls validate_uv_texture_map         # when uv_texture_map is set
    └── # invariants: faces index vertices; vertex_color rows == V; vertex_uv/face_uvs both-or-neither; face_uvs shape == faces shape; face_uvs index vertex_uv; vertex_color and uv_texture_map mutually exclusive; vertices/faces are the canonical geometry domain (U >= V)
```

## UV-origin convention

```text
data/structures/three_d/mesh/conventions.py
└── def transform_vertex_uv_convention(vertex_uv, source_convention, target_convention) -> torch.Tensor
    ├── # Transforms a UV table between origin conventions ("obj" = v from bottom; "top_left" = v from top).
    ├── calls validate_vertex_uv
    ├── calls validate_mesh_uv_convention                     # source and target
    └── # identity when conventions match; otherwise flips the V axis (v -> 1 - v).
```

## Loading

```text
data/structures/three_d/mesh/load.py
├── from pytorch3d.io import load_obj
├── from data.structures.three_d.mesh.merge import merge_meshes, pack_texture_images
├── def load_mesh(path) -> Dict                               # returns Mesh.__init__ kwargs
│   ├── # Loads one OBJ file or every OBJ under a mesh-root directory, then merges.
│   ├── calls _resolve_input_paths
│   ├── calls _load_mesh_attributes_from_obj_path             # per OBJ block
│   ├── calls merge_meshes
│   └── calls _mesh_to_init_kwargs
├── def _load_mesh_attributes_from_obj_path(obj_path) -> Dict
│   ├── # Dispatches one OBJ to the texture-representation-specific loader.
│   ├── calls _inspect_obj_file
│   ├── calls _load_mesh_uv_texture_map                       # has vt + uv faces
│   ├── calls _load_mesh_vertex_color                         # has v-line RGB
│   └── calls _load_mesh_geometry_only                        # otherwise
├── def _mesh_to_init_kwargs(mesh) -> Dict                    # Mesh -> detached/cpu/contiguous kwargs
├── def _load_mesh_geometry_only(path) -> Dict                # parses v / f lines
│   └── calls _resolve_input_path
├── def _load_mesh_vertex_color(path) -> Dict                 # parses v-with-RGB / f lines; >1 -> /255
│   └── calls _resolve_input_path
├── def _load_mesh_uv_texture_map(path) -> Dict               # PyTorch3D load_obj keeps the decoupled geometry/UV domains; convention "obj"
│   ├── calls _resolve_input_path
│   ├── calls load_obj                                        # verts, faces, aux (verts_uvs, textures_idx, texture_images)
│   └── calls pack_texture_images                             # multi-material -> single atlas
├── def _resolve_input_path(path) -> Path                     # exactly-one-OBJ resolution
│   └── calls _resolve_input_paths
├── def _resolve_input_paths(path) -> List[Path]              # OBJ file, or *.obj / */*.obj under a directory
└── def _inspect_obj_file(obj_path) -> Dict[str, bool]        # has_vertex_colors / has_uv_coords / has_uv_faces / has_mtllib
```

## Saving

```text
data/structures/three_d/mesh/save.py
├── def save_mesh(mesh, output_path) -> None
│   ├── # Dispatches one mesh to the texture-representation-specific writer.
│   ├── calls _save_mesh_vertex_color                         # when vertex_color is set
│   ├── calls _save_mesh_uv_texture_map                       # when uv_texture_map is set
│   └── calls _save_mesh_geometry_only                        # otherwise
├── def _save_mesh_geometry_only(mesh, output_path) -> None   # OBJ or PLY
│   └── calls _resolve_output_non_uv_mesh_path
├── def _save_mesh_vertex_color(mesh, output_path) -> None    # OBJ (v x y z r g b) or PLY
│   ├── calls _resolve_output_non_uv_mesh_path
│   ├── calls _normalize_vertex_color_for_obj                 # OBJ branch
│   └── calls _normalize_vertex_color_for_ply                 # PLY branch
├── def _save_mesh_uv_texture_map(mesh, output_path) -> None  # OBJ + sibling MTL + sibling texture PNG
│   ├── calls _resolve_output_obj_path
│   ├── calls _normalize_uv_texture_map_for_png
│   └── calls transform_vertex_uv_convention                  # mesh convention -> "obj" for the written vt lines
├── def _resolve_output_obj_path(output_path) -> Path         # ".obj" file, or "<dir>/mesh.obj"
├── def _resolve_output_non_uv_mesh_path(output_path) -> Path # ".obj"/".ply" file, or "<dir>/mesh.obj"
├── def _normalize_vertex_color_for_obj(vertex_color) -> torch.Tensor   # -> float32 [0,1]
├── def _normalize_vertex_color_for_ply(vertex_color) -> torch.Tensor   # -> uint8 [0,255]
└── def _normalize_uv_texture_map_for_png(uv_texture_map) -> np.ndarray # -> uint8 HWC
```

## Merging and texture-atlas packing

```text
data/structures/three_d/mesh/merge.py
├── def merge_meshes(mesh_blocks) -> Mesh
│   ├── # Merges one or more mesh blocks; pass-through for a single block.
│   ├── calls _merge_uv_textured_meshes                       # any block UV-textured
│   ├── calls _merge_vertex_color_meshes                      # any block vertex-colored
│   └── calls _merge_geometry_only_meshes                     # otherwise
├── def _merge_vertex_color_meshes(mesh_blocks) -> Mesh       # concat vertices/faces/colors with vertex offsets
├── def _merge_uv_textured_meshes(mesh_blocks) -> Mesh        # concat geometry + UV; pack per-block textures into one atlas
│   └── calls _pack_texture_maps
├── def _merge_geometry_only_meshes(mesh_blocks) -> Mesh      # concat vertices/faces with vertex offsets
├── def pack_texture_images(texture_images, verts_uvs, faces_uvs, materials_idx) -> Tuple
│   ├── # Public entry: packs a material-name -> image mapping into one atlas + remapped UVs.
│   └── calls _pack_texture_maps
├── def _pack_texture_maps(texture_maps, verts_uvs, faces_uvs, materials_idx) -> Tuple
│   ├── # Stacks texture maps vertically into one atlas; rebuilds per-corner UV table.
│   └── calls _remap_uvs
└── def _remap_uvs(verts_uvs, faces_uvs, map_offsets, atlas_height, atlas_width, materials_idx) -> torch.Tensor
    └── # Rescales/offsets each material's UVs into its packed atlas region.
```

## Framework interop conversions

```text
data/structures/three_d/mesh/convert.py
├── def mesh_to_pytorch3d(mesh, device=None, dtype=torch.float32) -> Meshes
│   ├── # Mesh -> PyTorch3D Meshes; TexturesVertex or TexturesUV; UV meshes forced to "obj" convention.
│   └── calls mesh.to
├── def mesh_from_pytorch3d(mesh, convention="obj") -> Mesh
│   └── # PyTorch3D Meshes -> Mesh; reads TexturesVertex / TexturesUV.
├── def mesh_from_open3d(mesh) -> Mesh                        # geometry + optional vertex colors (UV not supported)
├── def mesh_to_open3d(mesh) -> o3d.geometry.TriangleMesh     # geometry + optional vertex colors (UV not supported)
│   └── calls _mesh_vertex_color_to_float_rgb
├── def mesh_from_trimesh(mesh, convention=None) -> Mesh
│   ├── # trimesh.Trimesh -> Mesh; the UV branch welds per-corner duplicate vertices into the geometry domain.
│   ├── calls _weld_per_corner_uv_mesh                       # UV branch
│   ├── calls _normalize_trimesh_texture_image               # UV branch
│   └── calls _normalize_trimesh_vertex_colors               # vertex-color branch
├── def mesh_to_trimesh(mesh) -> trimesh.Trimesh
│   ├── # Mesh -> trimesh.Trimesh; UV meshes expanded to per-corner topology.
│   ├── calls _expand_obj_uv_mesh_for_trimesh                # UV branch
│   ├── calls _mesh_uv_texture_map_to_uint8                  # UV branch
│   └── calls _mesh_vertex_color_to_rgba                     # vertex-color branch
├── def _normalize_trimesh_texture_image(image) -> np.ndarray # -> uint8 HWC RGB (drops uniform alpha)
├── def _normalize_trimesh_vertex_colors(vertex_colors) -> np.ndarray  # -> float32/uint8 RGB (drops opaque alpha)
├── def _mesh_vertex_color_to_float_rgb(vertex_color) -> np.ndarray    # -> float32 RGB [0,1]
├── def _mesh_vertex_color_to_rgba(vertex_color) -> np.ndarray         # -> uint8 RGBA
├── def _mesh_uv_texture_map_to_uint8(uv_texture_map) -> np.ndarray    # -> uint8 HWC RGB
├── def _expand_obj_uv_mesh_for_trimesh(mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
│   └── # Expands an "obj"-convention UV mesh to per-corner topology (one vertex per v/vt pair) for trimesh.
└── def _weld_per_corner_uv_mesh(vertices, faces, vertex_uv) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    └── # Welds coincident per-corner duplicate vertices (exact-position equality) into the geometry domain; returns (vertices, faces, vertex_uv, face_uvs).
```

## Package API surface

```text
data/structures/three_d/mesh/__init__.py
└── re-exports: Mesh, load_mesh, save_mesh, merge_meshes, transform_vertex_uv_convention,
    mesh_from_open3d, mesh_from_pytorch3d, mesh_from_trimesh, mesh_to_open3d, mesh_to_pytorch3d, mesh_to_trimesh,
    validate_vertices, validate_faces, validate_vertex_color, validate_uv_texture_map,
    validate_vertex_uv, validate_face_uvs, validate_mesh_attributes, validate_mesh_uv_convention
```
