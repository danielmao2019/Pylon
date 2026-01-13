import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from torch import Tensor

from data.structures.three_d.camera.camera import Camera
from models.three_d.gspl.helpers import eval_sh
from models.three_d.gspl.model import GSPLModel

RendererOutputVisualizer = Callable[
    [torch.Tensor, Dict, "RendererOutputInfo"], torch.Tensor
]


@dataclass
class Camera:
    idx: Tensor
    R: Tensor  # [3, 3]
    T: Tensor  # [3]
    fx: Tensor
    fy: Tensor
    fov_x: Tensor
    fov_y: Tensor
    cx: Tensor
    cy: Tensor
    width: Tensor
    height: Tensor
    appearance_id: Tensor
    normalized_appearance_id: Tensor
    time: Tensor

    distortion_params: Optional[Tensor]
    """
    NOTE: this should be None or a zero tensor currently
        
    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor

    world_to_camera: Tensor
    projection: Tensor
    full_projection: Tensor
    camera_center: Tensor

    def to_device(self, device):
        for field in Camera.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))

        return self

    def get_K(self):
        K = torch.eye(4, dtype=torch.float, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy

        return K

    def get_full_perspective_projection(self):
        K = self.get_K()

        # full.transpose() = (K[R T]).transpose() = [R T].transpose() K.transpose()

        return self.world_to_camera @ K.T

    @property
    def device(self):
        return self.R.device


@dataclass
class Cameras:
    """
    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]

    distortion_params: Optional[Union[Tensor, List[Tensor]]]
    """
    NOTE: this should be None or zero tensors currently

    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor  # Int[n_cameras]

    world_to_camera: Tensor = field(init=False)  # [n_cameras, 4, 4], transposed
    projection: Tensor = field(init=False)
    full_projection: Tensor = field(init=False)
    camera_center: Tensor = field(init=False)

    time: Optional[Tensor] = None  # [n_cameras]

    idx: Tensor = None  # [N_cameras]

    def _calculate_fov(self):
        # calculate fov
        self.fov_x = 2 * torch.atan((self.width / 2) / self.fx)
        self.fov_y = 2 * torch.atan((self.height / 2) / self.fy)

    def _calculate_w2c(self):
        # build world-to-camera transform matrix
        self.world_to_camera = torch.zeros(
            size=(self.R.shape[0], 4, 4), dtype=self.R.dtype, device=self.R.device
        )
        self.world_to_camera[:, :3, :3] = self.R
        self.world_to_camera[:, :3, 3] = self.T
        self.world_to_camera[:, 3, 3] = 1.0
        self.world_to_camera = torch.transpose(self.world_to_camera, 1, 2)

    def _calculate_ndc_projection_matrix(self):
        """
        calculate ndc projection matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html

        TODO:
            1. support colmap refined principal points
            2. the near and far here are ignored in diff-gaussian-rasterization
        """
        zfar = 100.0
        znear = 0.01

        tanHalfFovY = torch.tan((self.fov_y / 2))
        tanHalfFovX = torch.tan((self.fov_x / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(
            size=(self.fov_y.shape[0], 4, 4),
            dtype=self.fov_y.dtype,
            device=self.fov_y.device,
        )

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * znear / (right - left)  # = 1 / tanHalfFovX = 2 * fx / width
        P[:, 1, 1] = 2.0 * znear / (top - bottom)  # = 2 * fy / height
        P[:, 0, 2] = (right + left) / (right - left)  # = 0, right + left = 0
        P[:, 1, 2] = (top + bottom) / (top - bottom)  # = 0, top + bottom = 0
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        self.projection = torch.transpose(P, 1, 2)

        self.full_projection = self.world_to_camera.bmm(self.projection)

    def _calculate_camera_center(self):
        self.camera_center = torch.linalg.inv(self.world_to_camera)[:, 3, :3]

    def __post_init__(self):
        self._calculate_fov()
        self._calculate_w2c()
        self._calculate_ndc_projection_matrix()
        self._calculate_camera_center()

        self.idx = torch.arange(self.R.shape[0], dtype=torch.int32)

        if self.time is None:
            self.time = torch.zeros(self.R.shape[0])
        if self.distortion_params is None:
            self.distortion_params = torch.zeros(self.R.shape[0], 4)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Camera:
        return Camera(
            idx=self.idx[index],
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            time=self.time[index],
            camera_type=self.camera_type[index],
            world_to_camera=self.world_to_camera[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class RendererOutputTypes:
    RGB: int = 1
    GRAY: int = 2
    NORMAL_MAP: int = 3
    FEATURE_MAP: int = 4
    OTHER: int = 65535  # must provide a visualizer


@dataclass
class RendererOutputInfo:
    key: str
    """The key used to retrieve value from the dictionary returned by `forward()`"""

    type: int = RendererOutputTypes.RGB
    """One defined in `RendererOutputTypes` above"""

    visualizer: RendererOutputVisualizer = None
    """
    The first parameter is the value retrieved from the dict returned by `forward()`. 
    The second parameter is the dict returned by `forward()`. 
    The Third one is a `RendererOutputInfo` instance.
    """

    def __post_init__(self):
        if self.type == RendererOutputTypes.OTHER and self.visualizer is None:
            raise ValueError("Visualizer must be provided when `type` is `OTHER`")


class Renderer(torch.nn.Module):

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GSPLModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list = None,
        **kwargs,
    ):
        pass

    def training_forward(
        self,
        step: int,
        module: Any,
        viewpoint_camera: Camera,
        pc: GSPLModel,
        bg_color: torch.Tensor,
        render_types: list = None,
        **kwargs,
    ):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            render_types=render_types,
            **kwargs,
        )

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_load_checkpoint(self, module, checkpoint):
        pass

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        pass

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return {"rgb": RendererOutputInfo("render")}


class VanillaRenderer(Renderer):

    def __init__(
        self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False
    ):
        super().__init__()

        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GSPLModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        render_types: list = None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if render_types is None:
            render_types = ["rgb"]
        assert len(render_types) == 1, "Only single type is allowed currently"

        rendered_image_key = "render"
        if "depth" in render_types:
            rendered_image_key = "depth"
            w2c = viewpoint_camera.world_to_camera  # already transposed
            means3D_in_camera_space = torch.matmul(pc.get_xyz, w2c[:3, :3]) + w2c[3, :3]
            depth = means3D_in_camera_space[:, 2:]
            # bg_color = torch.ones_like(bg_color) * depth.max()
            bg_color = torch.zeros_like(bg_color)
            override_color = depth.repeat(1, 3)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz,
                dtype=pc.get_xyz.dtype,
                requires_grad=True,
                device=bg_color.device,
            )
            + 0
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            rendered_image_key: rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth_image,
        }

    @staticmethod
    def render(
        means3D: torch.Tensor,  # xyz
        opacity: torch.Tensor,
        scales: Optional[torch.Tensor],
        rotations: Optional[torch.Tensor],
        features: Optional[torch.Tensor],  # shs
        active_sh_degree: int,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        colors_precomp: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
    ):
        if colors_precomp is not None:
            assert features is None
        if cov3D_precomp is not None:
            assert scales is None
            assert rotations is None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            means3D,
            dtype=means3D.dtype,
            requires_grad=True,
            device=means3D.device,
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rasterize_result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=features,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if len(rasterize_result) == 2:
            rendered_image, radii = rasterize_result
            depth_image = None
        else:
            rendered_image, radii, depth_image = rasterize_result

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY),
        }


class ViewerRenderer:

    def __init__(
        self,
        gaussian_model,
        renderer: Renderer,
        background_color,
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color

        self.max_depth = 0.0
        self.depth_map_color_map = "turbo"

        # TODO: initial value should get from renderer
        self.output_info: Tuple[str, RendererOutputInfo, RendererOutputVisualizer] = (
            "rgb",
            RendererOutputInfo("render"),
            self.no_processing,
        )

    def set_output_info(
        self,
        name: str,
        renderer_output_info: RendererOutputInfo,
        visualizer: RendererOutputVisualizer,
    ):
        self.output_info = (
            name,
            renderer_output_info,
            visualizer,
        )

    def _setup_depth_map_options(self, viewer, server):
        self.max_depth_gui_number = server.gui.add_number(
            label="Max Clamp",
            initial_value=self.max_depth,
            min=0.0,
            step=0.01,
            hint="value=0 means that no max clamping, value will be normalized based on the maximum one",
            visible=False,
        )
        self.depth_map_color_map_dropdown = server.gui.add_dropdown(
            label="Color Map",
            options=["turbo", "viridis", "magma", "inferno", "cividis", "gray"],
            initial_value=self.depth_map_color_map,
            visible=False,
        )

        @self.max_depth_gui_number.on_update
        @self.depth_map_color_map_dropdown.on_update
        def _(event):
            with server.atomic():
                self.max_depth = self.max_depth_gui_number.value
                self.depth_map_color_map = self.depth_map_color_map_dropdown.value
                viewer.rerender_for_all_client()

    def _set_depth_map_option_visibility(self, visible: bool):
        if getattr(self, "max_depth_gui_number", None) is None:
            return
        self.max_depth_gui_number.visible = visible
        self.depth_map_color_map_dropdown.visible = visible

    def _set_output_type(self, name: str, renderer_output_info: RendererOutputInfo):
        """
        Update properties
        """
        # toggle depth map option, only enable when type is `gray` and `visualizer` is None
        self._set_depth_map_option_visibility(
            renderer_output_info.type == RendererOutputTypes.GRAY
            and renderer_output_info.visualizer is None
        )

        # set visualizer
        visualizer = renderer_output_info.visualizer
        if visualizer is None:
            if renderer_output_info.type == RendererOutputTypes.RGB:
                visualizer = self.no_processing
            else:
                raise ValueError(
                    f"Unsupported output type `{renderer_output_info.type}`"
                )

        # update
        self.set_output_info(name, renderer_output_info, visualizer)

    def setup_options(self, viewer, server):
        available_outputs = self.renderer.get_available_outputs()
        first_type_name = list(available_outputs.keys())[0]

        with server.gui.add_folder("Output"):
            # setup output type dropdown
            output_type_dropdown = server.gui.add_dropdown(
                label="Type",
                options=list(available_outputs.keys()),
                initial_value=first_type_name,
            )
            self.output_type_dropdown = output_type_dropdown

            @output_type_dropdown.on_update
            def _(event):
                if event.client is None:
                    return
                with server.atomic():
                    # whether valid type
                    output_type_info = available_outputs.get(
                        output_type_dropdown.value, None
                    )
                    if output_type_info is None:
                        return

                    self._set_output_type(output_type_dropdown.value, output_type_info)

                    viewer.rerender_for_all_client()

            self._setup_depth_map_options(viewer, server)

        # update default output type to the first one, must be placed after gui setup
        self._set_output_type(
            name=first_type_name,
            renderer_output_info=available_outputs[first_type_name],
        )

    def get_outputs(self, camera, scaling_modifier: float = 1.0):
        render_type, output_info, output_processor = self.output_info

        render_outputs = self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
            render_types=[render_type],
        )
        image = output_processor(
            render_outputs[output_info.key], render_outputs, output_info
        )
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image

    def no_processing(self, i, *args, **kwargs):
        return i


def _prepare_camera(
    camera: Camera,
):
    """
    Build a single Camera from intrinsics/extrinsics.
    Assumes CUDA is available.
    """
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert torch.cuda.is_available(), "CUDA is required for rendering."
    device = torch.device("cuda")
    camera = camera.to(device=device, convention="opencv")
    extrinsics = camera.extrinsics.to(dtype=torch.float32)
    w2c = torch.linalg.inv(extrinsics)
    R = w2c[:3, :3].unsqueeze(0)
    T = w2c[:3, 3].unsqueeze(0)

    width = (
        torch.tensor([camera.cx * 2.0], device=device, dtype=torch.float32)
        .round()
        .clamp(min=1.0)
    )
    height = (
        torch.tensor([camera.cy * 2.0], device=device, dtype=torch.float32)
        .round()
        .clamp(min=1.0)
    )
    assert torch.all(width > 0) and torch.all(
        height > 0
    ), "Image dimensions must be positive"

    return Cameras(
        R=R,
        T=T,
        fx=torch.tensor([camera.fx], device=device, dtype=torch.float32),
        fy=torch.tensor([camera.fy], device=device, dtype=torch.float32),
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
        appearance_id=torch.zeros(1, dtype=torch.long, device=device),
        normalized_appearance_id=torch.zeros(1, dtype=torch.float, device=device),
        distortion_params=None,
        camera_type=torch.zeros(1, dtype=torch.long, device=device),
    )[0]


@torch.no_grad()
def render_rgb_from_matrices(
    model: GSPLModel,
    camera: Camera,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    scaling_modifier: float = 1.0,
) -> torch.Tensor:
    """
    Render an RGB tensor given a Gaussian model plus Camera and background color.
    A fresh VanillaRenderer is instantiated internally.
    """
    assert torch.cuda.is_available(), "CUDA is required for rendering."
    device = torch.device("cuda")
    assert isinstance(camera, Camera), f"{type(camera)=}"

    renderer = VanillaRenderer().to(device)
    renderer.setup("validation")
    model = model.to(device)
    model.eval()

    bg = torch.as_tensor(background_color, dtype=torch.float32, device=device).flatten()
    if bg.numel() == 1:
        bg = bg.expand(3)
    if bg.numel() != 3:
        raise ValueError("`background_color` must broadcast to 3 values (RGB).")

    outputs = renderer.get_available_outputs()
    if "rgb" not in outputs:
        raise ValueError("Renderer must expose an 'rgb' output.")

    viewer_renderer = ViewerRenderer(model, renderer, bg)
    viewer_renderer._set_output_type("rgb", outputs["rgb"])

    cam = _prepare_camera(
        camera=camera,
    )
    return viewer_renderer.get_outputs(cam, scaling_modifier=scaling_modifier)
