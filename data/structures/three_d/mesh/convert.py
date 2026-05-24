"""Class-conversion helpers for repo mesh objects."""

from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes

from data.structures.three_d.mesh.load import _shift_seam_crossing_faces_to_seam_safe
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.save import _collapse_seam_shifted_uv_rows
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def mesh_from_open3d(mesh: o3d.geometry.TriangleMesh) -> Mesh:
    """Convert one legacy Open3D triangle mesh into one Mesh.

    Args:
        mesh: Legacy Open3D triangle mesh carrying geometry and optional
            per-vertex colors (UV textures are not supported).

    Returns:
        Repo `Mesh` with geometry and an optional `MeshTextureVertexColor`.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, o3d.geometry.TriangleMesh), (
            "Expected `mesh` to be an Open3D `TriangleMesh`. " f"{type(mesh)=}"
        )
        assert not mesh.has_triangle_uvs(), (
            "Open3D UV-mesh conversion is not implemented in "
            "`mesh_from_open3d(...)`. "
            f"{mesh.has_triangle_uvs()=}"
        )
        assert not mesh.has_textures(), (
            "Open3D texture conversion is not implemented in "
            "`mesh_from_open3d(...)`. "
            f"{mesh.has_textures()=}"
        )

    _validate_inputs()

    verts_np = np.asarray(mesh.vertices)
    assert verts_np.dtype == np.float64, (
        "Expected Open3D vertex positions to be float64. " f"{verts_np.dtype=}"
    )
    faces_np = np.asarray(mesh.triangles)
    assert faces_np.dtype == np.int32, (
        "Expected Open3D triangle indices to be int32. " f"{faces_np.dtype=}"
    )

    texture: Optional[MeshTextureVertexColor] = None
    if mesh.has_vertex_colors():
        vertex_color_np = np.asarray(mesh.vertex_colors)
        assert vertex_color_np.dtype == np.float64, (
            "Expected Open3D vertex colors to be float64. " f"{vertex_color_np.dtype=}"
        )
        texture = MeshTextureVertexColor(
            vertex_color=torch.as_tensor(vertex_color_np, dtype=torch.float32)
        )

    return Mesh(
        verts=torch.as_tensor(verts_np, dtype=torch.float32),
        faces=torch.as_tensor(faces_np, dtype=torch.int64),
        texture=texture,
    )


def mesh_to_open3d(mesh: Mesh) -> o3d.geometry.TriangleMesh:
    """Convert one Mesh into one legacy Open3D triangle mesh.

    Args:
        mesh: Repo `Mesh` instance whose texture is `None` or a
            `MeshTextureVertexColor` (UV textures are not supported).

    Returns:
        Legacy Open3D triangle mesh with geometry and optional vertex colors.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert not isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Open3D export for UV-textured repo meshes is not implemented in "
            "`mesh_to_open3d(...)`. "
            f"{type(mesh.texture)=}"
        )

    _validate_inputs()

    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(
        mesh.verts.detach().cpu().numpy()
    )
    open3d_mesh.triangles = o3d.utility.Vector3iVector(
        mesh.faces.detach().cpu().numpy()
    )
    if isinstance(mesh.texture, MeshTextureVertexColor):
        vertex_color_np = _vertex_color_to_float_rgb(
            vertex_color=mesh.texture.vertex_color
        )
        open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            vertex_color_np.astype(np.float64)
        )
    return open3d_mesh


def mesh_from_pytorch3d(mesh: Meshes, convention: str = "obj") -> Mesh:
    """Convert one single-mesh PyTorch3D Meshes into one Mesh.

    Args:
        mesh: Single-mesh PyTorch3D `Meshes` container.
        convention: UV-origin convention to assign when UV textures are
            present.

    Returns:
        Repo `Mesh` carrying the same geometry and supported textures.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Meshes), (
            "Expected `mesh` to be a PyTorch3D `Meshes` instance. " f"{type(mesh)=}"
        )
        assert len(mesh) == 1, (
            "Expected `mesh_from_pytorch3d(...)` to receive exactly one mesh. "
            f"{len(mesh)=}"
        )
        assert isinstance(convention, str), (
            "Expected `convention` to be a string. " f"{type(convention)=}"
        )

    _validate_inputs()

    verts = mesh.verts_list()[0].to(dtype=torch.float32).contiguous()
    faces = mesh.faces_list()[0].to(dtype=torch.int64).contiguous()
    textures = mesh.textures
    if textures is None:
        return Mesh(verts=verts, faces=faces, texture=None)

    if isinstance(textures, TexturesVertex):
        vertex_color = textures.verts_features_list()[0].to(dtype=torch.float32)
        return Mesh(
            verts=verts,
            faces=faces,
            texture=MeshTextureVertexColor(vertex_color=vertex_color.contiguous()),
        )

    assert isinstance(textures, TexturesUV), (
        "Expected PyTorch3D mesh textures to be `None`, `TexturesVertex`, or "
        f"`TexturesUV`. {type(textures)=}"
    )
    raw_verts_uvs = textures.verts_uvs_list()[0].to(dtype=torch.float32).contiguous()
    raw_faces_uvs = textures.faces_uvs_list()[0].to(dtype=torch.int64).contiguous()
    canonical_verts_uvs, canonical_faces_uvs = _shift_seam_crossing_faces_to_seam_safe(
        verts_uvs=raw_verts_uvs,
        faces_uvs=raw_faces_uvs,
    )
    return Mesh(
        verts=verts,
        faces=faces,
        texture=MeshTextureUVTextureMap(
            uv_texture_map=textures.maps_padded()[0]
            .to(dtype=torch.float32)
            .contiguous(),
            verts_uvs=canonical_verts_uvs.contiguous(),
            faces_uvs=canonical_faces_uvs.contiguous(),
            convention=convention,
        ),
    )


def mesh_to_pytorch3d(
    mesh: Mesh,
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
) -> Meshes:
    """Convert one Mesh into one single-mesh PyTorch3D Meshes.

    Args:
        mesh: Repo `Mesh` instance.
        device: Optional target device for the PyTorch3D tensors.
        dtype: Target floating-point dtype for vertex and texture tensors.

    Returns:
        A single PyTorch3D `Meshes` instance.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected `device` to be `None`, a `str`, or a `torch.device`. "
            f"{type(device)=}"
        )
        assert isinstance(dtype, torch.dtype), (
            "Expected `dtype` to be a `torch.dtype`. " f"{type(dtype)=}"
        )

    _validate_inputs()

    target_device = mesh.device if device is None else torch.device(device)
    target_mesh = (
        mesh.to(device=target_device, convention="obj")
        if isinstance(mesh.texture, MeshTextureUVTextureMap)
        else mesh.to(device=target_device)
    )
    verts = target_mesh.verts.to(dtype=dtype).contiguous()
    faces = target_mesh.faces.to(dtype=torch.int64).contiguous()

    textures = None
    if isinstance(target_mesh.texture, MeshTextureVertexColor):
        textures = TexturesVertex(
            verts_features=[
                target_mesh.texture.vertex_color.to(dtype=dtype).contiguous()
            ]
        )
    elif isinstance(target_mesh.texture, MeshTextureUVTextureMap):
        obj_verts_uvs, obj_faces_uvs = _collapse_seam_shifted_uv_rows(
            verts_uvs=target_mesh.texture.verts_uvs.detach().cpu(),
            faces_uvs=target_mesh.texture.faces_uvs.detach().cpu(),
        )
        textures = TexturesUV(
            maps=[target_mesh.texture.uv_texture_map.to(dtype=dtype).contiguous()],
            faces_uvs=[
                obj_faces_uvs.to(device=target_device, dtype=torch.int64).contiguous()
            ],
            verts_uvs=[
                obj_verts_uvs.to(device=target_device, dtype=dtype).contiguous()
            ],
        )

    return Meshes(verts=[verts], faces=[faces], textures=textures)


def mesh_from_trimesh(mesh: trimesh.Trimesh, convention: Optional[str] = None) -> Mesh:
    """Convert one trimesh.Trimesh into one Mesh.

    When the trimesh carries UV data it is welded from trimesh's per-corner
    topology back onto the canonical geometry domain.

    Args:
        mesh: Source `trimesh.Trimesh` instance.
        convention: Required UV-origin convention when the trimesh carries UV
            data; `None` is accepted only for non-UV trimeshes.

    Returns:
        Repo `Mesh` with geometry and supported texture attributes.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, trimesh.Trimesh), (
            "Expected `mesh` to be a `trimesh.Trimesh`. " f"{type(mesh)=}"
        )
        assert convention is None or isinstance(convention, str), (
            "Expected `convention` to be `None` or a string. " f"{type(convention)=}"
        )

    _validate_inputs()

    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        assert convention is not None, (
            "Expected textured trimesh conversion to receive an explicit UV "
            f"`convention`. {convention=}"
        )
        assert mesh.visual.uv is not None, (
            "Expected a textured trimesh to carry UV coordinates. " f"{mesh.visual.uv=}"
        )
        verts, faces, verts_uvs, faces_uvs = _uv_mesh_from_trimesh(
            verts=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            verts_uvs=np.asarray(mesh.visual.uv),
        )
        canonical_verts_uvs, canonical_faces_uvs = _shift_seam_crossing_faces_to_seam_safe(
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
        )
        texture_image = _texture_image_from_trimesh(image=mesh.visual.material.image)
        return Mesh(
            verts=verts,
            faces=faces,
            texture=MeshTextureUVTextureMap(
                uv_texture_map=torch.as_tensor(texture_image),
                verts_uvs=canonical_verts_uvs,
                faces_uvs=canonical_faces_uvs,
                convention=convention,
            ),
        )

    verts_np = np.asarray(mesh.vertices)
    assert verts_np.dtype == np.float64, (
        "Expected trimesh vertex positions to be float64. " f"{verts_np.dtype=}"
    )
    faces_np = np.asarray(mesh.faces)
    assert faces_np.dtype == np.int64, (
        "Expected trimesh face indices to be int64. " f"{faces_np.dtype=}"
    )

    texture: Optional[MeshTextureVertexColor] = None
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        vertex_color = _vertex_color_from_trimesh(
            vertex_colors=np.asarray(mesh.visual.vertex_colors)
        )
        texture = MeshTextureVertexColor(vertex_color=torch.as_tensor(vertex_color))

    return Mesh(
        verts=torch.as_tensor(verts_np, dtype=torch.float32),
        faces=torch.as_tensor(faces_np, dtype=torch.int64),
        texture=texture,
    )


def mesh_to_trimesh(mesh: Mesh) -> trimesh.Trimesh:
    """Convert one Mesh into one trimesh.Trimesh.

    Args:
        mesh: Repo `Mesh` instance.

    Returns:
        `trimesh.Trimesh` with geometry and supported texture attributes.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )

    _validate_inputs()

    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        obj_mesh = mesh.to(convention="obj")
        assert isinstance(obj_mesh.texture, MeshTextureUVTextureMap), (
            "Expected the OBJ-convention mesh to keep a UV-texture-map texture. "
            f"{type(obj_mesh.texture)=}"
        )
        expanded_verts, expanded_faces, expanded_uv = _uv_mesh_to_trimesh(
            mesh=obj_mesh
        )
        texture_image = _texture_image_to_trimesh(
            uv_texture_map=obj_mesh.texture.uv_texture_map
        )
        return trimesh.Trimesh(
            vertices=expanded_verts,
            faces=expanded_faces,
            visual=trimesh.visual.texture.TextureVisuals(
                uv=expanded_uv,
                image=Image.fromarray(texture_image),
            ),
            process=False,
        )

    if isinstance(mesh.texture, MeshTextureVertexColor):
        return trimesh.Trimesh(
            vertices=mesh.verts.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            vertex_colors=_vertex_color_to_trimesh(
                vertex_color=mesh.texture.vertex_color
            ),
            process=False,
        )

    return trimesh.Trimesh(
        vertices=mesh.verts.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )


def _vertex_color_to_float_rgb(vertex_color: torch.Tensor) -> np.ndarray:
    """Convert one repo vertex-color tensor to a float32 RGB [0,1] array.

    Args:
        vertex_color: Repo vertex-color tensor in uint8 `[0, 255]` or float32
            `[0, 1]` form.

    Returns:
        Float32 RGB array in `[0, 1]`.
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_color, torch.Tensor), (
            "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(vertex_color)=}"
        )
        assert vertex_color.dtype in (torch.uint8, torch.float32), (
            "Expected repo vertex colors to use uint8 or float32 dtype. "
            f"{vertex_color.dtype=}"
        )

    _validate_inputs()

    vertex_color_cpu = vertex_color.detach().cpu()
    if vertex_color_cpu.dtype == torch.uint8:
        return vertex_color_cpu.to(dtype=torch.float32).div(255.0).numpy()
    return vertex_color_cpu.numpy()


def _uv_mesh_from_trimesh(
    verts: np.ndarray,
    faces: np.ndarray,
    verts_uvs: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weld trimesh's per-corner duplicate verts into the geometry domain.

    Trimesh's `force="mesh"` OBJ loader duplicates seam verts, producing one
    vertex per `v/vt` pair. This inverts that expansion: coincident positions
    (exact-position equality) are welded back to the canonical geometry domain,
    and the seam is re-expressed as `verts_uvs` / `faces_uvs`. It is the inverse
    of `_uv_mesh_to_trimesh`.

    Args:
        verts: Per-corner-expanded vertex positions `[N, 3]`.
        faces: Faces indexing the per-corner-expanded verts `[F, 3]`.
        verts_uvs: Per-corner UV coordinates `[N, 2]`, aligned 1:1 with
            `verts`.

    Returns:
        Tuple of welded geometry verts `[V, 3]` (float32), geometry faces
        `[F, 3]` (int64) indexing the welded verts, the UV table `[U, 2]`
        (float32), and `faces_uvs` `[F, 3]` (int64) indexing the UV table.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts, np.ndarray), (
            "Expected `verts` to be a numpy array. " f"{type(verts)=}"
        )
        assert verts.ndim == 2 and verts.shape[1] == 3, (
            "Expected `verts` to be `[N, 3]`. " f"{verts.shape=}"
        )
        assert isinstance(faces, np.ndarray), (
            "Expected `faces` to be a numpy array. " f"{type(faces)=}"
        )
        assert faces.ndim == 2 and faces.shape[1] == 3, (
            "Expected `faces` to be `[F, 3]`. " f"{faces.shape=}"
        )
        assert isinstance(verts_uvs, np.ndarray), (
            "Expected `verts_uvs` to be a numpy array. " f"{type(verts_uvs)=}"
        )
        assert verts_uvs.ndim == 2 and verts_uvs.shape[1] == 2, (
            "Expected `verts_uvs` to be `[N, 2]`. " f"{verts_uvs.shape=}"
        )
        assert verts_uvs.shape[0] == verts.shape[0], (
            "Expected per-corner trimesh UV to align 1:1 with verts before "
            "welding. "
            f"{verts_uvs.shape=} {verts.shape=}"
        )

    _validate_inputs()

    unique_positions, inverse_indices = np.unique(verts, axis=0, return_inverse=True)
    inverse_indices = inverse_indices.reshape(-1)
    welded_faces = inverse_indices[faces.reshape(-1)].reshape(faces.shape)
    return (
        torch.as_tensor(unique_positions, dtype=torch.float32).contiguous(),
        torch.as_tensor(welded_faces, dtype=torch.int64).contiguous(),
        torch.as_tensor(verts_uvs, dtype=torch.float32).contiguous(),
        torch.as_tensor(faces, dtype=torch.int64).contiguous(),
    )


def _texture_image_from_trimesh(image: Image.Image) -> np.ndarray:
    """Convert one trimesh material image into a uint8 HWC RGB array.

    Args:
        image: PIL texture image stored on a trimesh material.

    Returns:
        Uint8 HWC RGB texture image (uniform alpha dropped).
    """

    def _validate_inputs() -> None:
        assert isinstance(image, Image.Image), (
            "Expected the trimesh material image to be a PIL `Image`. "
            f"{type(image)=}"
        )

    _validate_inputs()

    image_np = np.asarray(image)
    assert image_np.ndim == 3 and image_np.shape[2] in (3, 4), (
        "Expected the trimesh texture image to be HWC with RGB or RGBA "
        "channels. "
        f"{image_np.shape=}"
    )
    assert image_np.dtype == np.uint8, (
        "Expected the trimesh texture image to be uint8. " f"{image_np.dtype=}"
    )
    if image_np.shape[2] == 4:
        alpha = image_np[:, :, 3]
        assert np.all(alpha == 255), (
            "Expected an RGBA trimesh texture to be fully opaque before "
            f"dropping its alpha channel. {np.unique(alpha)=}"
        )
        image_np = image_np[:, :, :3]
    return np.ascontiguousarray(image_np)


def _vertex_color_from_trimesh(vertex_colors: np.ndarray) -> np.ndarray:
    """Convert one trimesh vertex-color array into a repo RGB array.

    Args:
        vertex_colors: Trimesh vertex-color array `[V, 3]` or `[V, 4]`.

    Returns:
        Uint8 `[V, 3]` RGB array (opaque alpha dropped).
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_colors, np.ndarray), (
            "Expected `vertex_colors` to be a numpy array. " f"{type(vertex_colors)=}"
        )
        assert vertex_colors.ndim == 2 and vertex_colors.shape[1] in (3, 4), (
            "Expected `vertex_colors` to be `[V, 3]` or `[V, 4]`. "
            f"{vertex_colors.shape=}"
        )
        assert vertex_colors.dtype == np.uint8, (
            "Expected trimesh vertex colors to be uint8. " f"{vertex_colors.dtype=}"
        )

    _validate_inputs()

    if vertex_colors.shape[1] == 4:
        alpha = vertex_colors[:, 3]
        assert np.all(alpha == 255), (
            "Expected RGBA trimesh vertex colors to be fully opaque before "
            f"dropping their alpha channel. {np.unique(alpha)=}"
        )
        vertex_colors = vertex_colors[:, :3]
    return np.ascontiguousarray(vertex_colors)


def _uv_mesh_to_trimesh(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand one "obj"-convention UV mesh into trimesh's per-corner topology.

    It is the inverse of `_uv_mesh_from_trimesh`.

    Args:
        mesh: Repo UV mesh already normalized to the `"obj"` convention.

    Returns:
        Tuple of expanded verts `[3F, 3]`, expanded faces `[F, 3]`, and
        per-corner UV coordinates `[3F, 2]`.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Expected UV mesh expansion to receive a `MeshTextureUVTextureMap`. "
            f"{type(mesh.texture)=}"
        )
        assert mesh.texture.convention == "obj", (
            "Expected trimesh UV expansion to receive OBJ-convention UVs. "
            f"{mesh.texture.convention=}"
        )

    _validate_inputs()

    face_verts = mesh.faces.detach().cpu()
    obj_verts_uvs, obj_faces_uvs = _collapse_seam_shifted_uv_rows(
        verts_uvs=mesh.texture.verts_uvs.detach().cpu(),
        faces_uvs=mesh.texture.faces_uvs.detach().cpu(),
    )
    expanded_verts = mesh.verts.detach().cpu()[face_verts.reshape(-1)].numpy()
    expanded_uv = obj_verts_uvs[obj_faces_uvs.reshape(-1)].numpy()
    expanded_faces = np.arange(expanded_verts.shape[0], dtype=np.int64).reshape(
        -1, 3
    )
    return expanded_verts, expanded_faces, expanded_uv


def _texture_image_to_trimesh(uv_texture_map: torch.Tensor) -> np.ndarray:
    """Convert one repo uv_texture_map tensor into a uint8 HWC RGB array.

    Args:
        uv_texture_map: Repo UV texture map tensor in HWC RGB layout.

    Returns:
        Uint8 HWC RGB texture image.
    """

    def _validate_inputs() -> None:
        assert isinstance(uv_texture_map, torch.Tensor), (
            "Expected `uv_texture_map` to be a `torch.Tensor`. "
            f"{type(uv_texture_map)=}"
        )
        assert uv_texture_map.ndim == 3 and uv_texture_map.shape[2] == 3, (
            "Expected the repo UV texture map to use HWC RGB layout. "
            f"{uv_texture_map.shape=}"
        )
        assert uv_texture_map.dtype in (torch.uint8, torch.float32), (
            "Expected the repo UV texture map to use uint8 or float32 dtype. "
            f"{uv_texture_map.dtype=}"
        )

    _validate_inputs()

    texture_cpu = uv_texture_map.detach().cpu()
    if texture_cpu.dtype == torch.uint8:
        return np.ascontiguousarray(texture_cpu.numpy())
    return np.ascontiguousarray(
        texture_cpu.mul(255.0).round().to(dtype=torch.uint8).numpy()
    )


def _vertex_color_to_trimesh(vertex_color: torch.Tensor) -> np.ndarray:
    """Convert one repo vertex_color tensor into a uint8 RGBA array for trimesh.

    Args:
        vertex_color: Repo vertex-color tensor.

    Returns:
        Uint8 `[V, 4]` RGBA color array.
    """

    rgb_float = _vertex_color_to_float_rgb(vertex_color=vertex_color)
    rgb_uint8 = np.rint(np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = np.full((rgb_uint8.shape[0], 1), fill_value=255, dtype=np.uint8)
    return np.concatenate([rgb_uint8, alpha], axis=1)
