"""Class-conversion helpers for repo mesh objects."""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes

from data.structures.three_d.mesh.validate import validate_mesh_uv_convention

if TYPE_CHECKING:
    from data.structures.three_d.mesh.mesh import Mesh


def mesh_to_pytorch3d(
    mesh: "Mesh",
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
) -> Meshes:
    """Convert one repo mesh into one PyTorch3D mesh.

    Args:
        mesh: Repo mesh instance.
        device: Optional target device for the PyTorch3D tensors.
        dtype: Target floating-point dtype for vertex and texture tensors.

    Returns:
        A single PyTorch3D `Meshes` instance.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

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
        if mesh.uv_texture_map is not None
        else mesh.to(device=target_device)
    )
    verts = target_mesh.vertices.to(dtype=dtype).contiguous()
    faces = target_mesh.faces.to(dtype=torch.int64).contiguous()

    textures = None
    if target_mesh.vertex_color is not None:
        textures = TexturesVertex(
            verts_features=[target_mesh.vertex_color.to(dtype=dtype).contiguous()]
        )
    elif target_mesh.uv_texture_map is not None:
        textures = TexturesUV(
            maps=[target_mesh.uv_texture_map.to(dtype=dtype).contiguous()],
            faces_uvs=[target_mesh.face_uvs.to(dtype=torch.int64).contiguous()],
            verts_uvs=[target_mesh.vertex_uv.to(dtype=dtype).contiguous()],
        )

    return Meshes(verts=[verts], faces=[faces], textures=textures)


def mesh_from_pytorch3d(mesh: Meshes, convention: str = "obj") -> "Mesh":
    """Convert one PyTorch3D mesh into one repo mesh.

    Args:
        mesh: Single-mesh PyTorch3D `Meshes` container.
        convention: UV-origin convention to assign when UV textures are present.

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

    from data.structures.three_d.mesh.mesh import Mesh

    vertices = mesh.verts_list()[0].to(dtype=torch.float32).contiguous()
    faces = mesh.faces_list()[0].to(dtype=torch.int64).contiguous()
    textures = mesh.textures
    if textures is None:
        return Mesh(vertices=vertices, faces=faces)

    if isinstance(textures, TexturesVertex):
        vertex_color = textures.verts_features_list()[0].to(dtype=torch.float32)
        return Mesh(
            vertices=vertices,
            faces=faces,
            vertex_color=vertex_color.contiguous(),
        )

    assert isinstance(textures, TexturesUV), (
        "Expected PyTorch3D mesh textures to be `None`, `TexturesVertex`, or "
        f"`TexturesUV`. {type(textures)=}"
    )
    validate_mesh_uv_convention(convention=convention)
    return Mesh(
        vertices=vertices,
        faces=faces,
        uv_texture_map=textures.maps_padded()[0].to(dtype=torch.float32).contiguous(),
        vertex_uv=textures.verts_uvs_list()[0].to(dtype=torch.float32).contiguous(),
        face_uvs=textures.faces_uvs_list()[0].to(dtype=torch.int64).contiguous(),
        convention=convention,
    )


def mesh_from_open3d(mesh: o3d.geometry.TriangleMesh) -> "Mesh":
    """Convert one legacy Open3D triangle mesh into one repo mesh.

    Args:
        mesh: Legacy Open3D triangle mesh.

    Returns:
        Repo `Mesh` with geometry and optional vertex colors.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, o3d.geometry.TriangleMesh), (
            "Expected `mesh` to be an Open3D `TriangleMesh`. " f"{type(mesh)=}"
        )
        assert not getattr(mesh, "has_triangle_uvs", lambda: False)(), (
            "Open3D UV-mesh conversion is not implemented in `mesh_from_open3d(...)`. "
            f"{mesh.has_triangle_uvs()=}"
        )
        assert not getattr(mesh, "has_textures", lambda: False)(), (
            "Open3D texture conversion is not implemented in `mesh_from_open3d(...)`. "
            f"{mesh.has_textures()=}"
        )

    _validate_inputs()

    from data.structures.three_d.mesh.mesh import Mesh

    vertex_color = None
    if mesh.has_vertex_colors():
        vertex_color = torch.as_tensor(
            np.asarray(mesh.vertex_colors),
            dtype=torch.float32,
        )
    return Mesh(
        vertices=torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32),
        faces=torch.as_tensor(np.asarray(mesh.triangles), dtype=torch.int64),
        vertex_color=vertex_color,
    )


def mesh_to_open3d(mesh: "Mesh") -> o3d.geometry.TriangleMesh:
    """Convert one repo mesh into one legacy Open3D triangle mesh.

    Args:
        mesh: Repo `Mesh` instance.

    Returns:
        Legacy Open3D triangle mesh with geometry and optional vertex colors.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert mesh.uv_texture_map is None, (
            "Open3D export for UV-textured repo meshes is not implemented in "
            "`mesh_to_open3d(...)`. "
            f"{mesh.uv_texture_map is None=}"
        )

    _validate_inputs()

    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(
        mesh.vertices.detach().cpu().numpy()
    )
    open3d_mesh.triangles = o3d.utility.Vector3iVector(
        mesh.faces.detach().cpu().numpy()
    )
    if mesh.vertex_color is not None:
        color_np = _mesh_vertex_color_to_float_rgb(vertex_color=mesh.vertex_color)
        open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            color_np.astype(np.float64)
        )
    return open3d_mesh


def mesh_from_trimesh(
    mesh: trimesh.Trimesh, convention: Optional[str] = None
) -> "Mesh":
    """Convert one trimesh mesh into one repo mesh.

    Args:
        mesh: Source `trimesh.Trimesh` instance.
        convention: Required UV-origin convention when the trimesh carries UV data.

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

    from data.structures.three_d.mesh.mesh import Mesh

    vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64)
    texture_uv = getattr(mesh.visual, "uv", None)
    if texture_uv is not None:
        assert convention is not None, (
            "Expected textured trimesh conversion to receive an explicit UV "
            f"`convention`. {convention=}"
        )
        validate_mesh_uv_convention(convention=convention)
        texture_image = _normalize_trimesh_texture_image(
            image=getattr(getattr(mesh.visual, "material", None), "image", None)
        )
        return Mesh(
            vertices=vertices,
            faces=faces,
            uv_texture_map=torch.as_tensor(texture_image),
            vertex_uv=torch.as_tensor(np.asarray(texture_uv), dtype=torch.float32),
            face_uvs=faces.clone(),
            convention=convention,
        )

    vertex_color = None
    trimesh_vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if trimesh_vertex_colors is not None and len(trimesh_vertex_colors) == len(
        mesh.vertices
    ):
        vertex_color = torch.as_tensor(
            _normalize_trimesh_vertex_colors(
                vertex_colors=np.asarray(trimesh_vertex_colors)
            ),
        )
    return Mesh(vertices=vertices, faces=faces, vertex_color=vertex_color)


def mesh_to_trimesh(mesh: "Mesh") -> trimesh.Trimesh:
    """Convert one repo mesh into one trimesh mesh.

    Args:
        mesh: Repo `Mesh` instance.

    Returns:
        `trimesh.Trimesh` with geometry and supported texture attributes.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )

    _validate_inputs()

    vertices_np = mesh.vertices.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    if mesh.uv_texture_map is not None:
        obj_mesh = mesh.to(convention="obj")
        expanded_vertices_np, expanded_faces_np, expanded_uv_np = (
            _expand_obj_uv_mesh_for_trimesh(mesh=obj_mesh)
        )
        texture_image = _mesh_uv_texture_map_to_uint8(
            uv_texture_map=obj_mesh.uv_texture_map
        )
        visual = trimesh.visual.texture.TextureVisuals(
            uv=expanded_uv_np,
            image=Image.fromarray(texture_image),
        )
        return trimesh.Trimesh(
            vertices=expanded_vertices_np,
            faces=expanded_faces_np,
            visual=visual,
            process=False,
        )

    if mesh.vertex_color is not None:
        return trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_colors=_mesh_vertex_color_to_rgba(vertex_color=mesh.vertex_color),
            process=False,
        )
    return trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)


def _normalize_trimesh_texture_image(image: object) -> np.ndarray:
    """Normalize one trimesh texture image to HWC RGB form.

    Args:
        image: Texture image object from trimesh material storage.

    Returns:
        Uint8 HWC RGB texture image.
    """

    assert image is not None, (
        "Expected textured trimesh conversion to receive a material image. "
        f"{type(image)=}"
    )
    image_np = np.asarray(image)
    assert image_np.ndim == 3, (
        "Expected trimesh texture image to be rank 3. " f"{image_np.shape=}"
    )
    assert image_np.shape[2] in (3, 4), (
        "Expected trimesh texture image to have RGB or RGBA channels. "
        f"{image_np.shape=}"
    )
    if image_np.shape[2] == 4:
        alpha = image_np[:, :, 3]
        assert np.all(alpha == alpha.reshape(-1)[0]), (
            "Expected RGBA trimesh textures to use a uniform alpha channel before "
            f"dropping alpha for repo `Mesh`. {np.unique(alpha)=}"
        )
        if np.issubdtype(alpha.dtype, np.floating):
            assert float(alpha.reshape(-1)[0]) in (0.0, 1.0), (
                "Expected floating RGBA trimesh texture alpha to be 0 or 1 before "
                f"dropping it. {float(alpha.reshape(-1)[0])=}"
            )
        else:
            assert int(alpha.reshape(-1)[0]) in (0, 255), (
                "Expected integer RGBA trimesh texture alpha to be 0 or 255 before "
                f"dropping it. {int(alpha.reshape(-1)[0])=}"
            )
        image_np = image_np[:, :, :3]
    if np.issubdtype(image_np.dtype, np.floating):
        image_np = np.clip(image_np, 0.0, 1.0)
        return np.rint(image_np * 255.0).astype(np.uint8)
    return image_np.astype(np.uint8, copy=False)


def _normalize_trimesh_vertex_colors(vertex_colors: np.ndarray) -> np.ndarray:
    """Normalize trimesh vertex colors to repo RGB form.

    Args:
        vertex_colors: Trimesh vertex-color array.

    Returns:
        Float32 or uint8 RGB array accepted by repo `Mesh`.
    """

    assert isinstance(vertex_colors, np.ndarray), (
        "Expected `vertex_colors` to be a numpy array. " f"{type(vertex_colors)=}"
    )
    assert vertex_colors.ndim == 2, (
        "Expected `vertex_colors` to be rank 2. " f"{vertex_colors.shape=}"
    )
    assert vertex_colors.shape[1] in (3, 4), (
        "Expected trimesh vertex colors to have RGB or RGBA channels. "
        f"{vertex_colors.shape=}"
    )
    if vertex_colors.shape[1] == 4:
        alpha = vertex_colors[:, 3]
        if np.issubdtype(alpha.dtype, np.floating):
            assert np.all((alpha == 0.0) | (alpha == 1.0)), (
                "Expected floating RGBA trimesh vertex alpha to be 0 or 1 before "
                f"dropping it. {np.unique(alpha)=}"
            )
            assert np.all(alpha == 1.0), (
                "Expected RGBA trimesh vertex colors to be opaque before dropping "
                f"alpha for repo `Mesh`. {np.unique(alpha)=}"
            )
        else:
            assert np.all((alpha == 0) | (alpha == 255)), (
                "Expected integer RGBA trimesh vertex alpha to be 0 or 255 before "
                f"dropping it. {np.unique(alpha)=}"
            )
            assert np.all(alpha == 255), (
                "Expected RGBA trimesh vertex colors to be opaque before dropping "
                f"alpha for repo `Mesh`. {np.unique(alpha)=}"
            )
        vertex_colors = vertex_colors[:, :3]
    if np.issubdtype(vertex_colors.dtype, np.floating):
        return vertex_colors.astype(np.float32, copy=False)
    return vertex_colors.astype(np.uint8, copy=False)


def _mesh_vertex_color_to_float_rgb(vertex_color: torch.Tensor) -> np.ndarray:
    """Normalize one repo vertex-color tensor to float RGB values.

    Args:
        vertex_color: Repo vertex-color tensor.

    Returns:
        Float32 RGB array in `[0, 1]`.
    """

    assert isinstance(vertex_color, torch.Tensor), (
        "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(vertex_color)=}"
    )
    vertex_color_cpu = vertex_color.detach().cpu()
    if vertex_color_cpu.dtype == torch.uint8:
        return vertex_color_cpu.to(dtype=torch.float32).div(255.0).numpy()
    assert vertex_color_cpu.dtype == torch.float32, (
        "Expected repo vertex colors to use uint8 or float32 dtype. "
        f"{vertex_color_cpu.dtype=}"
    )
    return vertex_color_cpu.numpy()


def _mesh_vertex_color_to_rgba(vertex_color: torch.Tensor) -> np.ndarray:
    """Normalize one repo vertex-color tensor to RGBA uint8 values for trimesh.

    Args:
        vertex_color: Repo vertex-color tensor.

    Returns:
        Uint8 RGBA color array.
    """

    rgb_float = _mesh_vertex_color_to_float_rgb(vertex_color=vertex_color)
    rgb_uint8 = np.rint(np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = np.full((rgb_uint8.shape[0], 1), fill_value=255, dtype=np.uint8)
    return np.concatenate([rgb_uint8, alpha], axis=1)


def _mesh_uv_texture_map_to_uint8(uv_texture_map: torch.Tensor) -> np.ndarray:
    """Normalize one repo UV texture map to uint8 HWC form.

    Args:
        uv_texture_map: Repo UV texture map tensor.

    Returns:
        Uint8 HWC RGB texture image.
    """

    assert isinstance(uv_texture_map, torch.Tensor), (
        "Expected `uv_texture_map` to be a `torch.Tensor`. " f"{type(uv_texture_map)=}"
    )
    texture_cpu = uv_texture_map.detach().cpu()
    assert texture_cpu.ndim == 3 and texture_cpu.shape[2] == 3, (
        "Expected repo UV texture map to use HWC RGB layout. " f"{texture_cpu.shape=}"
    )
    if texture_cpu.dtype == torch.uint8:
        return texture_cpu.numpy()
    assert texture_cpu.dtype == torch.float32, (
        "Expected repo UV texture map to use uint8 or float32 dtype. "
        f"{texture_cpu.dtype=}"
    )
    return texture_cpu.mul(255.0).round().to(dtype=torch.uint8).numpy()


def _expand_obj_uv_mesh_for_trimesh(
    mesh: "Mesh",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand one OBJ-convention repo UV mesh to per-corner UV topology for trimesh.

    Args:
        mesh: Repo UV mesh already normalized to `"obj"` convention.

    Returns:
        Tuple of expanded vertices, faces, and per-vertex UV coordinates.
    """

    from data.structures.three_d.mesh.mesh import Mesh

    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert mesh.vertex_uv is not None and mesh.face_uvs is not None, (
        "Expected UV mesh expansion to receive UV topology. "
        f"{mesh.vertex_uv is not None=} {mesh.face_uvs is not None=}"
    )
    assert mesh.convention == "obj", (
        "Expected trimesh UV expansion to receive OBJ-convention UVs. "
        f"{mesh.convention=}"
    )

    face_vertices = mesh.faces.detach().cpu()
    face_uvs = mesh.face_uvs.detach().cpu()
    expanded_vertices = mesh.vertices.detach().cpu()[face_vertices.reshape(-1)].numpy()
    expanded_uv = mesh.vertex_uv.detach().cpu()[face_uvs.reshape(-1)].numpy()
    expanded_faces = np.arange(expanded_vertices.shape[0], dtype=np.int64).reshape(
        -1, 3
    )
    return expanded_vertices, expanded_faces, expanded_uv
