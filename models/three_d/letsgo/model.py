import os

import numpy as np
import torch
from plyfile import PlyData
from torch import nn


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class _LetsGoModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, level: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.level = level
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree


class LetsGoModel:

    def __init__(self, sh_degree: int, max_level: int, depth_max: float, beta: float):
        self.max_level = max_level
        self.depth_max = depth_max
        self.beta = beta
        self.gaussians = [
            _LetsGoModel(sh_degree, level) for level in range(max_level + 1)
        ]

    def load_ply(self, iter_dir: str):
        for level in range(self.max_level + 1):
            ply_path = f"{iter_dir}/level_{level}.ply"
            assert os.path.isfile(ply_path), f"Missing LetsGo level file at {ply_path}"
            self.gaussians[level].load_ply(ply_path)

    def get_z_depth(self, xyz, viewmatrix):
        homogeneous_xyz = torch.cat(
            (xyz, torch.ones(xyz.shape[0], 1, dtype=xyz.dtype, device=xyz.device)),
            dim=1,
        )
        projected_xyz = torch.matmul(homogeneous_xyz, viewmatrix)
        depth_z = projected_xyz[:, 2]
        return depth_z

    def get_gaussian_parameters(
        self, viewpoint, compute_cov3D_python, scaling_modifier=1.0, random=-1
    ):

        levels = range(self.max_level + 1)
        get_attrs = lambda attr: [
            getattr(self.gaussians[level], attr) for level in levels
        ]
        xyz, features, opacity, scales, rotations = map(
            get_attrs,
            ['get_xyz', 'get_features', 'get_opacity', 'get_scaling', 'get_rotation'],
        )

        # Compute cov3D_precomp if necessary
        cov3D_precomp = (
            [self.gaussians[-1].get_covariance(scaling_modifier)] * len(xyz)
            if compute_cov3D_python
            else None
        )

        # Define activation levels based on 'random' parameter
        if random < 0:
            depths = [self.get_z_depth(xyz_lvl.detach(), viewpoint) for xyz_lvl in xyz]
            act_levels = [
                torch.clamp(
                    (self.max_level + 1)
                    * torch.exp(-1.0 * self.beta * torch.abs(depth) / self.depth_max),
                    0,
                    self.max_level,
                )
                for depth in depths
            ]
            act_levels = [torch.floor(level) for level in act_levels]
            filters = [
                act_level == level for act_level, level in zip(act_levels, levels)
            ]
        else:
            filters = [
                torch.full_like(xyz[level][:, 0], level == random, dtype=torch.bool)
                for level in levels
            ]

        # Concatenate all attributes
        concat_attrs = lambda attrs: torch.cat(attrs, dim=0)
        xyz, features, opacity, scales, rotations, filters = map(
            concat_attrs, [xyz, features, opacity, scales, rotations, filters]
        )

        # Apply filters to all attributes
        filtered = lambda attr: attr[filters]
        xyz, features, opacity, scales, rotations = map(
            filtered, [xyz, features, opacity, scales, rotations]
        )

        if compute_cov3D_python:
            cov3D_precomp = filtered(concat_attrs(cov3D_precomp))

        # Active and maximum spherical harmonics degrees
        active_sh_degree, max_sh_degree = (
            self.gaussians[-1].active_sh_degree,
            self.gaussians[-1].max_sh_degree,
        )

        return (
            xyz,
            features,
            opacity,
            scales,
            rotations,
            cov3D_precomp,
            active_sh_degree,
            max_sh_degree,
            filters,
        )
