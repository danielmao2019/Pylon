from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    device = r.device
    dtype = r.dtype
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=device, dtype=dtype)
    r0 = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r0 * z)
    R[:, 0, 2] = 2 * (x * z + r0 * y)
    R[:, 1, 0] = 2 * (x * y + r0 * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r0 * x)
    R[:, 2, 0] = 2 * (x * z - r0 * y)
    R[:, 2, 1] = 2 * (y * z + r0 * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    device = L.device
    dtype = L.dtype
    out = torch.zeros((L.shape[0], 6), dtype=dtype, device=device)
    out[:, 0] = L[:, 0, 0]
    out[:, 1] = L[:, 0, 1]
    out[:, 2] = L[:, 0, 2]
    out[:, 3] = L[:, 1, 1]
    out[:, 4] = L[:, 1, 2]
    out[:, 5] = L[:, 2, 2]
    return out


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    return strip_lowerdiag(sym)


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    device = s.device
    dtype = s.dtype
    L = torch.zeros((s.shape[0], 3, 3), dtype=dtype, device=device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


class Cheng2025CLODGS:
    """Gaussian model with per-splat LoD sigma for Cheng et al. 2025 CLOD (render-only)."""

    def setup_functions(self) -> None:
        def build_covariance_from_scaling_rotation(
            scaling: torch.Tensor, scaling_modifier: float, rotation: torch.Tensor
        ) -> torch.Tensor:
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, optimizer_type: str = "default") -> None:
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._lod_sigma = torch.empty(0)
        self.setup_functions()

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._features_rest

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_lod_sigma(self) -> torch.Tensor:
        return torch.relu(self._lod_sigma)

    def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def load_ply(
        self, path: Union[str, Path], device: Union[str, torch.device] = "cuda"
    ) -> None:
        ply_path = Path(path)
        assert ply_path.is_file(), f"PLY file not found at {ply_path}"
        device = torch.device(device)

        plydata = PlyData.read(str(ply_path))

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

        if "lod_sigma" in plydata.elements[0]:
            lod_sigma = np.asarray(plydata.elements[0]["lod_sigma"])[..., np.newaxis]
        else:
            lod_sigma = np.ones((xyz.shape[0], 1))

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True)
        )
        self._lod_sigma = nn.Parameter(
            torch.tensor(lod_sigma, dtype=torch.float, device=device).requires_grad_(
                True
            )
        )

        self.active_sh_degree = self.max_sh_degree

    def to(self, device: Union[str, torch.device]) -> "Cheng2025CLODGS":
        device = torch.device(device)
        tensor_attributes = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
            "_lod_sigma",
        ]

        for name in tensor_attributes:
            assert hasattr(self, name), f"Cheng2025CLODGS missing tensor '{name}'"
            value = getattr(self, name)
            if isinstance(value, nn.Parameter):
                if value.device != device:
                    value.data = value.data.to(device)
                    if value.grad is not None:
                        value.grad = value.grad.to(device)
            elif torch.is_tensor(value):
                if value.device != device:
                    setattr(self, name, value.to(device))

        return self
